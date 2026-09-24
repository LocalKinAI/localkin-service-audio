"""Qwen3-TTS via the official ``qwen-tts`` package (pins transformers 4.57)."""
from _protocol import pick_device, serve, write_wav

state = {}

LANGUAGES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
             "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
             "es": "Spanish", "it": "Italian"}


def load(repo, params=None):
    import torch
    from qwen_tts import Qwen3TTSModel

    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    state["model"] = Qwen3TTSModel.from_pretrained(repo, device_map=device, dtype=dtype)
    state["kind"] = ("voicedesign" if "VoiceDesign" in repo
                     else "base" if "-Base" in repo else "customvoice")
    return {"device": device}


def synthesize(text, out, voice=None, language=None, instruct=None,
               ref_audio=None, ref_text=None, **_):
    m, kind = state["model"], state["kind"]
    lang = LANGUAGES.get(language or "", "Auto")
    if kind == "customvoice":
        speakers = {s.lower(): s for s in (m.get_supported_speakers() or [])}
        voice = speakers.get((voice or "vivian").lower(), voice or "Vivian")
        wavs, sr = m.generate_custom_voice(text=text, language=lang, speaker=voice,
                                           instruct=instruct or None)
    elif kind == "voicedesign":
        wavs, sr = m.generate_voice_design(text=text, language=lang,
                                           instruct=instruct or "A clear, natural voice.")
    else:
        wavs, sr = m.generate_voice_clone(text=text, language=lang, ref_audio=ref_audio, ref_text=ref_text)
    return write_wav(out, wavs[0], sr)


serve({"load": load, "synthesize": synthesize})
