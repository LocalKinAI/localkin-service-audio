"""FireRedASR2 (AED) via the official FireRedASR2S repo (Python >= 3.11)."""
from _protocol import serve

state = {}


def load(repo, params=None):
    import torch
    from huggingface_hub import snapshot_download
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    cfg = FireRedAsr2Config(use_gpu=torch.cuda.is_available(), use_half=False,
                            beam_size=3, nbest=1, return_timestamp=True)
    state["model"] = FireRedAsr2.from_pretrained("aed", snapshot_download(repo), cfg)
    return {}


def transcribe(audio, language=None, **_):
    import numpy as np
    import soundfile as sf
    import librosa

    wav, sr = sf.read(audio, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    res = state["model"].transcribe(["utt"], [(16000, pcm)])[0]
    # Timestamp layout isn't documented; take (token, start, end) triples
    # when that's what comes back and otherwise return text only.
    segments = []
    try:
        for t in res.get("timestamp") or []:
            segments.append({"start": float(t[1]), "end": float(t[2]), "text": str(t[0])})
    except (TypeError, ValueError, IndexError, KeyError):
        segments = []
    return {"text": res.get("text", ""), "segments": segments, "language": "zh"}


serve({"load": load, "transcribe": transcribe})
