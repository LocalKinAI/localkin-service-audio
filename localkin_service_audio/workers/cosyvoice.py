"""CosyVoice 2 / Fun-CosyVoice3 from the official repo (no pip package).

The environment clones FunAudioLLM/CosyVoice; LOCALKIN_COSYVOICE_REPO points
at it so ``cosyvoice`` and the bundled Matcha-TTS import.
"""
import os
import sys

from _protocol import serve, write_wav

state = {}
_repo_dir = os.environ.get("LOCALKIN_COSYVOICE_REPO", "")
sys.path[:0] = [_repo_dir, os.path.join(_repo_dir, "third_party", "Matcha-TTS")]


def load(repo, params=None):
    from huggingface_hub import snapshot_download
    from cosyvoice.cli.cosyvoice import AutoModel

    state["model"] = AutoModel(model_dir=snapshot_download(repo))
    state["v3"] = "CosyVoice3" in repo
    return {}


def synthesize(text, out, voice=None, ref_audio=None, ref_text=None, instruct=None, **_):
    import numpy as np

    m = state["model"]
    prefix = "You are a helpful assistant.<|endofprompt|>" if state["v3"] else ""
    speakers = m.list_available_spks() if hasattr(m, "list_available_spks") else []
    if speakers and (voice in speakers or not ref_audio):
        # v1 SFT models ship named speakers (中文女, 英文男, ...)
        chunks = m.inference_sft(text, voice if voice in speakers else speakers[0], stream=False)
    elif instruct:
        tag = instruct if instruct.endswith("<|endofprompt|>") else instruct + "<|endofprompt|>"
        chunks = m.inference_instruct2(text, ("You are a helpful assistant. " if state["v3"] else "") + tag,
                                       ref_audio, stream=False)
    elif ref_text:
        chunks = m.inference_zero_shot(text, prefix + ref_text, ref_audio, stream=False)
    else:
        chunks = m.inference_cross_lingual(prefix + text, ref_audio, stream=False)
    wav = np.concatenate([c["tts_speech"].squeeze(0).cpu().numpy() for c in chunks])
    return write_wav(out, wav, m.sample_rate)


serve({"load": load, "synthesize": synthesize})
