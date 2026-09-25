"""Tests for ComfyUI-backed music: blueprint conversion and the HTTP round trip.

The blueprint here is a small hand-made one with the same shape as ComfyUI's
"Text to Music (MiniMax Music 3)": a subgraph whose inputs feed widget
inputs, a seed with the UI's extra control value after it, and one AUDIO
output. A tiny local HTTP server stands in for ComfyUI.
"""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import soundfile as sf

from localkin_service_audio.core.types import ModelConfig, ModelType
from localkin_service_audio.music import comfyui_strategy as cs
from localkin_service_audio.music.comfyui_blueprint import (
    BlueprintError, missing_files, subgraph_inputs, to_api_prompt,
)

OBJECT_INFO = {
    "Loader": {"input": {"required": {"unet_name": [["music.safetensors"]]}},
               "input_order": {"required": ["unet_name"]}, "output": ["MODEL"]},
    "Encode": {"input": {"required": {"model": ["MODEL"], "caption": ["STRING", {}], "lyrics": ["STRING", {}],
                                      "seed": ["INT", {"control_after_generate": True}],
                                      "max_duration": ["FLOAT", {}], "cfg_scale": ["FLOAT", {}]}},
               "input_order": {"required": ["model", "caption", "lyrics", "seed", "max_duration", "cfg_scale"]},
               "output": ["AUDIO"]},
}

BLUEPRINT = {
    "nodes": [{"id": 9, "type": "sg-uuid", "widgets_values": []}],
    "definitions": {"subgraphs": [{
        "name": "Text to Music (Test)",
        "inputs": [{"name": "caption"}, {"name": "lyrics"}, {"name": "max_duration"}, {"name": "seed"}],
        "outputs": [{"name": "AUDIO"}],
        "nodes": [
            {"id": 1, "type": "Loader", "widgets_values": ["music.safetensors"],
             "inputs": [{"name": "unet_name", "widget": {"name": "unet_name"}}]},
            # caption, lyrics, seed, <control value>, max_duration, cfg_scale
            {"id": 2, "type": "Encode", "widgets_values": ["", "", 222, "fixed", 60, 1.7],
             "inputs": [{"name": "model"}, {"name": "caption"}, {"name": "lyrics"}, {"name": "seed"},
                        {"name": "max_duration"}, {"name": "cfg_scale"}]},
            {"id": 3, "type": "Note", "widgets_values": ["ignore me"]},
        ],
        "links": [
            [1, 1, 0, 2, 0, "MODEL"],
            [2, -10, 0, 2, 1, "STRING"],   # caption
            [3, -10, 1, 2, 2, "STRING"],   # lyrics
            [4, -10, 3, 2, 3, "INT"],      # seed
            [5, -10, 2, 2, 4, "FLOAT"],    # max_duration
            [6, 2, 0, -20, 0, "AUDIO"],
        ],
    }]},
}


def test_converts_widgets_links_and_output():
    graph, save = to_api_prompt(BLUEPRINT, OBJECT_INFO, {"caption": "lofi", "seed": 7})
    enc = graph["2"]["inputs"]
    assert enc["model"] == ["1", 0]
    assert enc["caption"] == "lofi" and enc["seed"] == 7
    # not supplied: the blueprint's own values, with "fixed" skipped
    assert enc["lyrics"] == "" and enc["max_duration"] == 60 and enc["cfg_scale"] == 1.7
    assert all(n["class_type"] != "Note" for n in graph.values())   # UI-only, dropped
    assert graph[save] == {"class_type": "SaveAudio",
                           "inputs": {"audio": ["2", 0], "filename_prefix": "localkin/music"}}
    assert subgraph_inputs(BLUEPRINT) == ["caption", "lyrics", "max_duration", "seed"]


def test_missing_model_files_are_named():
    graph, _ = to_api_prompt(BLUEPRINT, OBJECT_INFO, {})
    assert missing_files(graph, OBJECT_INFO) == []
    info = json.loads(json.dumps(OBJECT_INFO))
    info["Loader"]["input"]["required"]["unet_name"] = [["other.safetensors"]]
    assert missing_files(graph, info) == ["music.safetensors"]


def test_disabled_dead_end_nodes_are_dropped():
    bp = json.loads(json.dumps(BLUEPRINT))
    sg = bp["definitions"]["subgraphs"][0]
    sg["nodes"].append({"id": 5, "type": "PreviewAny", "mode": 4, "inputs": [{"name": "source"}]})
    sg["links"].append([7, 2, 0, 5, 0, "AUDIO"])   # reads from Encode, feeds nothing
    graph, _ = to_api_prompt(bp, OBJECT_INFO, {})
    assert all(n["class_type"] != "PreviewAny" for n in graph.values())
    sg["nodes"][0]["mode"] = 4                        # the Loader feeds Encode
    with pytest.raises(BlueprintError, match="feeds other nodes"):
        to_api_prompt(bp, OBJECT_INFO, {})


def test_unknown_node_type_is_a_clear_error():
    info = {k: v for k, v in OBJECT_INFO.items() if k != "Encode"}
    with pytest.raises(BlueprintError, match="Encode"):
        to_api_prompt(BLUEPRINT, info, {})


class FakeComfy(BaseHTTPRequestHandler):
    queued = []
    fail = None

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/global_subgraphs":
            self._json({"k1": {"name": "Text to Music (Test)"}})
        elif self.path == "/api/global_subgraphs/k1":
            self._json({"name": "Text to Music (Test)", "data": json.dumps(BLUEPRINT)})
        elif self.path == "/object_info":
            self._json(OBJECT_INFO)
        elif self.path.startswith("/history/"):
            if FakeComfy.fail:
                self._json({"p1": {"status": {"status_str": "error", "messages": [[FakeComfy.fail, {}]]}}})
            else:
                graph = FakeComfy.queued[-1]["prompt"]
                save = next(k for k, n in graph.items() if n["class_type"] == "SaveAudio")
                self._json({"p1": {"status": {"completed": True},
                                   "outputs": {save: {"audio": [{"filename": "a.flac", "subfolder": "localkin",
                                                                 "type": "output"}]}}}})
        elif self.path.startswith("/view?"):
            buf = io.BytesIO()
            sf.write(buf, np.zeros((4800, 2), dtype=np.float32), 48000, format="FLAC")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(buf.getvalue())
        else:
            self.send_error(404)

    def do_POST(self):
        FakeComfy.queued.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self._json({"prompt_id": "p1"})


@pytest.fixture
def comfy():
    FakeComfy.queued, FakeComfy.fail = [], None
    server = HTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _engine(url):
    e = cs.ComfyUIMusicStrategy(url)
    assert e.load(ModelConfig(name="comfyui:Text to Music (Test)", type=ModelType.TTS, engine="music")), e.load_error
    return e


def test_generate_round_trip(comfy):
    r = _engine(comfy).generate("rainy lofi", duration=30, lyrics="[Verse] hi", seed=3, tags="piano")
    assert r.sample_rate == 48000 and r.audio.shape == (4800, 2)
    enc = FakeComfy.queued[-1]["prompt"]["2"]["inputs"]
    assert enc["caption"] == "rainy lofi, piano"     # no tags input: folded into caption
    assert enc["lyrics"] == "[Verse] hi" and enc["max_duration"] == 30 and enc["seed"] == 3


def test_interrupt_is_reported_as_such(comfy):
    e = _engine(comfy)
    FakeComfy.fail = "execution_interrupted"
    with pytest.raises(RuntimeError, match="interrupted"):
        e.generate("x")


def test_unreachable_comfyui_explains_itself():
    e = cs.ComfyUIMusicStrategy("http://127.0.0.1:9")
    assert e.load(ModelConfig(name="minimax-music3", type=ModelType.TTS, engine="music")) is False
    assert "can't reach ComfyUI" in e.load_error


def test_unknown_blueprint_lists_audio_ones(comfy):
    e = cs.ComfyUIMusicStrategy(comfy)
    assert e.load(ModelConfig(name="yue2", type=ModelType.TTS, engine="music")) is False
    assert "Text to Music (Test)" in e.load_error
