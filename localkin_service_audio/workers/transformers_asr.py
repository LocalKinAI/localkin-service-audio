"""Speech recognition through transformers >= 5.17 (CUDA / CPU / MPS).

Only Whisper, Parakeet and Nemotron work through ``pipeline()``; the
audio-LLM families (Qwen3-ASR, Fun-ASR-Nano, GLM-ASR, VibeVoice-ASR, Canary,
Voxtral) each need their processor's own request format. FAMILIES records
the differences so the three code paths below stay generic.
"""
from _protocol import pick_device, serve

state = {}

FAMILIES = {
    "whisper": {"kind": "pipeline"},
    "parakeet_tdt": {"kind": "transducer", "timestamps": True},
    "parakeet_rnnt": {"kind": "transducer", "timestamps": True},
    "parakeet_ctc": {"kind": "transducer", "timestamps": False},
    "nemotron_asr_streaming": {"kind": "transducer", "timestamps": True},
    # A bare "zh" is rejected; it wants locales.
    "nemotron3_5_asr": {"kind": "transducer", "timestamps": True, "lang_kw": "language",
                        "lang_default": "auto", "lang_map": {"zh": "zh-CN", "ja": "ja-JP", "yue": "zh-HK"}},
    "qwen3_asr": {"kind": "request", "lang_kw": "language", "strip_prompt": True,
                  "decode_kw": {"return_format": "parsed"}},
    "fun_asr_nano": {"kind": "request", "lang_kw": "language", "lang_allowed": {"zh", "en", "ja"},
                     "strip_prompt": True, "decode_kw": {"strip_prefix": True}},
    "glmasr": {"kind": "request", "strip_prompt": True},
    "vibevoice_asr": {"kind": "request", "sample_rate": 24000, "strip_prompt": True,
                      "decode_kw": {"return_format": "parsed"}},
    # Canary has no auto-detect and no Chinese; default to English.
    "canary": {"kind": "request", "lang_kw": "source_language", "lang_default": "en"},
    "voxtral": {"kind": "request", "lang_kw": "language", "strip_prompt": True, "voxtral": True},
    "voxtral_realtime": {"kind": "call"},
}


def load(repo, params=None):
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    device = pick_device()
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    if cfg.model_type not in FAMILIES:
        raise ValueError(f"unsupported ASR family {cfg.model_type!r}")
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)
    model_class = getattr(transformers, cfg.architectures[0])
    model = model_class.from_pretrained(repo, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(repo)
    state.update(repo=repo, device=device, model=model, processor=processor,
                 family=cfg.model_type, spec=FAMILIES[cfg.model_type])
    if state["spec"]["kind"] == "pipeline":
        state["pipe"] = transformers.pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor, device=device)
    return {"device": device, "family": cfg.model_type}


def _audio(path, sample_rate):
    import librosa
    return librosa.load(path, sr=sample_rate, mono=True)[0]


def _language(spec, language):
    if not language or language == "auto":
        return spec.get("lang_default")
    if "lang_allowed" in spec and language not in spec["lang_allowed"]:
        return spec.get("lang_default")
    return spec.get("lang_map", {}).get(language, language)


def transcribe(audio, language=None, **_):
    import torch

    spec, proc, model, device = state["spec"], state["processor"], state["model"], state["device"]
    lang = _language(spec, language)
    wav = _audio(audio, spec.get("sample_rate", 16000))

    if spec["kind"] == "pipeline":
        r = state["pipe"]({"raw": wav, "sampling_rate": 16000}, return_timestamps=True,
                          generate_kwargs={"language": lang} if lang else {})
        segs = [{"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"]}
                for c in r.get("chunks") or [] if c["timestamp"][0] is not None and c["timestamp"][1] is not None]
        return {"text": r["text"], "segments": segs, "language": language}

    kw = {spec["lang_kw"]: lang} if spec.get("lang_kw") and lang else {}
    with torch.inference_mode():
        if spec["kind"] == "transducer":
            inputs = proc(wav, sampling_rate=16000, **kw).to(device, dtype=model.dtype)
            out = model.generate(**inputs, return_dict_in_generate=True)
            if spec["timestamps"]:
                texts, stamps = proc.decode(out.sequences, durations=out.durations, skip_special_tokens=True)
                segs = [{"start": t["start"], "end": t["end"], "text": t["token"]} for t in stamps[0]]
                return {"text": texts[0], "segments": _merge_tokens(segs), "language": language}
            return {"text": proc.batch_decode(out.sequences, skip_special_tokens=True)[0], "language": language}

        if spec["kind"] == "request":
            extra = {"model_id": state["repo"], "sampling_rate": 16000, "format": "wav"} if spec.get("voxtral") else {}
            inputs = proc.apply_transcription_request(wav, **kw, **extra)
        else:
            inputs = proc(wav, return_tensors="pt", sampling_rate=16000)
        inputs = inputs.to(device, dtype=model.dtype)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=4096)
        if spec.get("strip_prompt"):
            out = out[:, inputs["input_ids"].shape[1]:]
        decoded = proc.decode(out, skip_special_tokens=True, **spec.get("decode_kw", {}))

    first = decoded[0] if isinstance(decoded, list) else decoded
    if isinstance(first, dict):  # qwen3_asr "parsed": {"language", "transcription"}
        return {"text": first.get("transcription", ""), "language": first.get("language") or language}
    if isinstance(first, list):  # vibevoice "parsed": [{"Start","End","Speaker","Content"}]
        segs = [{"start": s.get("Start"), "end": s.get("End"), "text": s.get("Content", ""),
                 "speaker": s.get("Speaker")} for s in first]
        return {"text": " ".join(s["text"] for s in segs), "segments": segs, "language": language}
    return {"text": str(first), "language": language}


def _merge_tokens(tokens, gap=0.6):
    """Subword timestamps -> phrase segments split on pauses."""
    segs = []
    for t in tokens:
        if segs and t["start"] - segs[-1]["end"] < gap:
            segs[-1]["end"] = t["end"]
            segs[-1]["text"] += t["text"]
        else:
            segs.append(dict(t))
    for s in segs:
        s["text"] = s["text"].replace("▁", " ").strip()
    return segs


serve({"load": load, "transcribe": transcribe})
