"""OmniVoice (k2-fsa) via the official ``omnivoice`` package, 24 kHz."""
from _protocol import pick_device, serve, write_wav

state = {}


def load(repo, params=None):
    import torch
    from omnivoice import OmniVoice

    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    state["model"] = OmniVoice.from_pretrained(repo, device_map=device, dtype=dtype)
    return {"device": device}


def synthesize(text, out, ref_audio=None, ref_text=None, **_):
    wavs = state["model"].generate(text, ref_audio=ref_audio, ref_text=ref_text)
    return write_wav(out, wavs[0], 24000)


serve({"load": load, "synthesize": synthesize})
