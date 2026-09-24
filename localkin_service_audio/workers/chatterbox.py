"""Chatterbox (Resemble AI) via ``chatterbox-tts`` (pins torch 2.6, numpy<2)."""
from _protocol import pick_device, serve, write_wav

state = {}


def load(repo, params=None):
    device = pick_device()
    if "turbo" in repo.lower():
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
        state["multilingual"] = False
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
        state["multilingual"] = True
    state["model"] = Model.from_pretrained(device=device)
    return {"device": device}


def synthesize(text, out, language=None, ref_audio=None, exaggeration=None, cfg_weight=None, **_):
    m = state["model"]
    kwargs = {"audio_prompt_path": ref_audio}
    if exaggeration is not None:
        kwargs["exaggeration"] = exaggeration
    if cfg_weight is not None:
        kwargs["cfg_weight"] = cfg_weight
    if state["multilingual"]:
        kwargs["language_id"] = language or "en"
    wav = m.generate(text, **kwargs)
    return write_wav(out, wav.squeeze(0).cpu().numpy(), m.sr)


serve({"load": load, "synthesize": synthesize})
