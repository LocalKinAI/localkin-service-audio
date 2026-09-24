"""Speech synthesis through transformers >= 5.17: Sesame CSM, Dia, Higgs
Audio v2, VibeVoice. One small prompt builder per family; loading and
decoding are shared. The server always sends a reference voice (none of
these ship presets), so every call sounds the same.
"""
from _protocol import pick_device, serve, write_wav

state = {}

SAMPLE_RATES = {"csm": 24000, "dia": 44100, "higgs_audio_v2": 24000, "vibevoice": 24000}
FRAME_RATES = {"csm": 12.5, "dia": 86, "higgs_audio_v2": 25, "vibevoice": 7.5}


def load(repo, params=None):
    import torch
    from transformers import AutoConfig, AutoProcessor

    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    family = AutoConfig.from_pretrained(repo).model_type
    if family not in SAMPLE_RATES:
        raise ValueError(f"unsupported TTS family {family!r}")
    if family == "dia":  # seq2seq mapping, not text-to-waveform
        from transformers import DiaForConditionalGeneration as model_class
    else:
        from transformers import AutoModelForTextToWaveform as model_class
    model = model_class.from_pretrained(repo, dtype=dtype, device_map=device).eval()
    # The processor loads the codec; Higgs decodes wherever it lands.
    processor = AutoProcessor.from_pretrained(repo, device_map=device)
    state.update(device=device, dtype=dtype, family=family, model=model, processor=processor,
                 max_seconds=(params or {}).get("max_seconds", 30))
    return {"device": device, "family": family}


def _chat(conversation, **kw):
    return state["processor"].apply_chat_template(
        conversation, tokenize=True, return_dict=True, return_tensors="pt", **kw
    ).to(state["device"], state["dtype"])


def _csm(text, speaker, ref, ref_text, frames):
    conv = []
    if ref:
        conv.append({"role": speaker, "content": [{"type": "text", "text": ref_text or ""},
                                                  {"type": "audio", "path": ref}]})
    conv.append({"role": speaker, "content": [{"type": "text", "text": text}]})
    return state["model"].generate(**_chat(conv), output_audio=True, max_new_tokens=frames)[0]


def _dia(text, speaker, ref, ref_text, frames):
    import librosa

    proc = state["processor"]
    tag = lambda s: s if s.lstrip().startswith("[S") else f"[S1] {s}"
    prompt, audio = tag(text), None
    if ref:
        audio = [librosa.load(ref, sr=44100, mono=True)[0]]
        prompt = f"{tag(ref_text or '')} {prompt}"
    inputs = proc(text=[prompt], audio=audio, padding=True, return_tensors="pt").to(state["device"])
    out = state["model"].generate(**inputs, max_new_tokens=frames)
    plen = proc.get_audio_prompt_len(inputs["decoder_attention_mask"]) if ref else None
    return proc.batch_decode(out, audio_prompt_len=plen)[0]


def _higgs_audio_v2(text, speaker, ref, ref_text, frames):
    conv = [{"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
            {"role": "scene", "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]}]
    if ref:
        conv += [{"role": "user", "content": [{"type": "text", "text": ref_text or ""}]},
                 {"role": "assistant", "content": [{"type": "audio", "path": ref}]}]
    conv.append({"role": "user", "content": [{"type": "text", "text": text}]})
    codes = state["model"].generate(**_chat(conv, add_generation_prompt=True, sampling_rate=24000),
                                    max_new_tokens=frames)
    return state["processor"].decode(codes)


def _vibevoice(text, speaker, ref, ref_text, frames):
    content = [{"type": "text", "text": text}] + ([{"type": "audio", "path": ref}] if ref else [])
    # add_generation_prompt=True, or the reference is treated as the target.
    return state["model"].generate(**_chat([{"role": speaker, "content": content}], add_generation_prompt=True),
                                   max_new_tokens=frames)[0]


def synthesize(text, out, voice=None, ref_audio=None, ref_text=None, **_):
    import torch

    family = state["family"]
    frames = int(state["max_seconds"] * FRAME_RATES[family])
    with torch.inference_mode():
        wav = globals()[f"_{family}"](text, voice or "0", ref_audio, ref_text, frames)
    return write_wav(out, wav.detach().float().cpu().numpy(), SAMPLE_RATES[family])


serve({"load": load, "synthesize": synthesize})
