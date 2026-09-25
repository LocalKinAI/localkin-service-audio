"""
Turn a ComfyUI blueprint (a saved subgraph, as the UI stores it) into the API
"prompt" graph ComfyUI's /prompt endpoint executes.

ComfyUI ships its text-to-music workflows — MiniMax Music 3, ACE-Step,
YuE2, Stable Audio — as blueprints in the UI's subgraph format, which only the
browser front end knows how to run. This does the front end's conversion:

- widget values are matched to inputs in the order ComfyUI's own
  /object_info declares them, skipping the extra "control after generate"
  value the UI stores after seed-like inputs;
- links are resolved into ``[node_id, output_slot]`` references, with the
  subgraph's inputs replaced by values the caller supplies (falling back to
  the value saved in the blueprint);
- the subgraph's audio output is wired to a SaveAudio node.
"""
from typing import Any, Dict, List, Optional, Tuple

SUBGRAPH_INPUT = -10
SUBGRAPH_OUTPUT = -20

# UI-only nodes with no backend class.
_UI_ONLY = {"Note", "MarkdownNote", "PrimitiveNode"}

_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


class BlueprintError(ValueError):
    pass


def _links(subgraph: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = ["id", "origin_id", "origin_slot", "target_id", "target_slot", "type"]
    return [l if isinstance(l, dict) else dict(zip(keys, l)) for l in subgraph.get("links", [])]


def _is_widget(spec: Any) -> bool:
    kind = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if opts.get("forceInput"):
        return False
    return isinstance(kind, list) or kind in _WIDGET_TYPES


def _has_control_widget(name: str, spec: Any) -> bool:
    """The UI stores a "fixed"/"randomize" value after seed-like inputs."""
    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "control_after_generate" in opts:
        return bool(opts["control_after_generate"])
    kind = spec[0] if isinstance(spec, (list, tuple)) else spec
    return kind == "INT" and name in ("seed", "noise_seed")


def _widget_values(node: Dict[str, Any], object_info: Dict[str, Any]) -> Dict[str, Any]:
    info = object_info.get(node["type"])
    if info is None:
        raise BlueprintError(f"ComfyUI has no node type {node['type']!r} (update ComfyUI?)")
    values = node.get("widgets_values") or []
    if isinstance(values, dict):          # some nodes store them by name
        return dict(values)
    specs = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
    order = info.get("input_order") or {}
    names = (order.get("required", []) + order.get("optional", [])) or list(specs)
    out, i = {}, 0
    for name in names:
        spec = specs.get(name)
        if spec is None or not _is_widget(spec):
            continue
        if i >= len(values):
            break
        out[name] = values[i]
        i += 1
        if _has_control_widget(name, spec):
            i += 1
    return out


def subgraph_inputs(blueprint: Dict[str, Any]) -> List[str]:
    """Names the blueprint exposes (caption, lyrics, seed, ...)."""
    return [i["name"] for i in _subgraph(blueprint).get("inputs", [])]


def _subgraph(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    subgraphs = (blueprint.get("definitions") or {}).get("subgraphs") or []
    if not subgraphs:
        raise BlueprintError("not a subgraph blueprint")
    return subgraphs[0]


def to_api_prompt(
    blueprint: Dict[str, Any],
    object_info: Dict[str, Any],
    values: Dict[str, Any],
    filename_prefix: str = "localkin/music",
) -> Tuple[Dict[str, Any], str]:
    """Build the /prompt graph. Returns (graph, id of the SaveAudio node)."""
    sg = _subgraph(blueprint)
    nodes = {n["id"]: n for n in sg["nodes"] if n.get("type") not in _UI_ONLY}
    sg_inputs = [i["name"] for i in sg.get("inputs", [])]

    # Muted (2) and bypassed (4) nodes: drop the ones nothing reads from — a
    # disabled preview, say. A bypassed node feeding others would need its
    # inputs passed through, which isn't supported.
    feeds = {l["origin_id"] for l in _links(sg) if l["target_id"] in nodes or l["target_id"] == SUBGRAPH_OUTPUT}
    for nid, n in list(nodes.items()):
        if n.get("mode", 0) in (2, 4):
            if nid in feeds:
                raise BlueprintError(f"node {nid} ({n['type']}) is muted or bypassed but feeds other nodes")
            del nodes[nid]

    graph: Dict[str, Dict[str, Any]] = {}
    for nid, node in nodes.items():
        graph[str(nid)] = {"class_type": node["type"], "inputs": _widget_values(node, object_info)}

    output_source = None
    for link in _links(sg):
        origin, target = link["origin_id"], link["target_id"]
        if target == SUBGRAPH_OUTPUT:
            output_source = [str(origin), link["origin_slot"]]
            continue
        if target not in nodes:
            continue
        target_inputs = nodes[target].get("inputs", [])
        slot = link["target_slot"]
        if slot >= len(target_inputs):
            raise BlueprintError(f"link {link['id']} points past node {target}'s inputs")
        name = target_inputs[slot]["name"]
        if origin == SUBGRAPH_INPUT:
            sg_name = sg_inputs[link["origin_slot"]]
            if values.get(sg_name) is not None:
                graph[str(target)]["inputs"][name] = values[sg_name]
            # else: keep the widget value saved in the blueprint
        elif origin in nodes:
            graph[str(target)]["inputs"][name] = [str(origin), link["origin_slot"]]

    if output_source is None:
        raise BlueprintError("blueprint has no output")
    save_id = str(max(int(k) for k in graph) + 1)
    graph[save_id] = {"class_type": "SaveAudio",
                      "inputs": {"audio": output_source, "filename_prefix": filename_prefix}}
    return graph, save_id


def missing_files(graph: Dict[str, Any], object_info: Dict[str, Any]) -> List[str]:
    """Model files the graph names that ComfyUI doesn't have (combo inputs
    whose value isn't among the options ComfyUI lists)."""
    missing = []
    for node in graph.values():
        info = object_info.get(node["class_type"]) or {}
        specs = {**info.get("input", {}).get("required", {}), **info.get("input", {}).get("optional", {})}
        for name, value in node["inputs"].items():
            spec = specs.get(name)
            if not isinstance(value, str) or not spec:
                continue
            options = spec[0] if isinstance(spec[0], list) else (spec[1] or {}).get("options") if len(spec) > 1 else None
            if isinstance(options, list) and value.endswith((".safetensors", ".ckpt", ".pt", ".bin", ".gguf")) \
                    and value not in options:
                missing.append(value)
    return missing
