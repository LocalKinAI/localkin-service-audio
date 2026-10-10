"""Audio8 ASR Infinite via its official package (github.com/Edge0-AI/Audio8-ASR-Infinite).

A streaming model: its simulated-streaming decoder walks the clip one text
token per 80 ms, transcribing 480 ms behind the audio. It knows Chinese and
English and has to be told which; without a language it gets Chinese.
"""
from _protocol import pick_device, serve

state = {}

SAMPLE_RATE = 16000
TOKEN_MS = 80
DELAY_MS = 480


class _StreamConfig:
    """The audio settings the decoder reads (duck-typed, as in the package's example)."""
    sampling_rate = SAMPLE_RATE
    raw_audio_samples_per_token = SAMPLE_RATE * TOKEN_MS // 1000

    def __init__(self, left_pad_tokens):
        self.streaming_n_left_pad_tokens = left_pad_tokens


def load(repo, params=None):
    import torch
    from transformers import Qwen2Tokenizer, VoxtralRealtimeFeatureExtractor
    from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
        Audio8ASRInfiniteForConditionalGeneration, resolve_qwen_streaming_special_token_ids,
    )

    device = pick_device()
    # The checkpoint is bf16 and the package's own example runs it that way everywhere.
    dtype = torch.bfloat16
    # The classes the repo's configs name, not Auto*: those read config.json's
    # auto_map and stop to ask on stdin whether to run the repo's own code.
    tokenizer = Qwen2Tokenizer.from_pretrained(repo)
    # The semantic VAD heads live in a second weights file that from_pretrained
    # never reads (model.safetensors wins over the index), and its exact-match
    # check then refuses the model. Transcription doesn't use them: skip them.
    config = Audio8ASRInfiniteForConditionalGeneration.config_class.from_pretrained(repo)
    config.semantic_vad_horizons_seconds = None
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        repo, config=config, dtype=dtype).to(device).eval()
    model.config.use_cache = True
    state.update(
        model=model, tokenizer=tokenizer, dtype=dtype,
        feature_extractor=VoxtralRealtimeFeatureExtractor.from_pretrained(repo),
        special_ids=resolve_qwen_streaming_special_token_ids(tokenizer),
        stream=_StreamConfig(int(getattr(model.config, "streaming_n_left_pad_tokens", 0) or 18)),
    )
    return {"device": device}


def transcribe(audio, language=None, **_):
    language = "zh" if language in (None, "", "auto") else language
    if language not in ("zh", "en"):
        raise ValueError(f"Audio8 ASR Infinite transcribes zh or en, got {language!r}")

    import librosa
    import numpy as np
    from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import resolve_qwen_language_token_id
    from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode_batch

    wav = librosa.load(audio, sr=SAMPLE_RATE, mono=True)[0]
    model = state["model"]
    result = simulated_streaming_greedy_decode_batch(
        model=model, tokenizer=state["tokenizer"], feature_extractor=state["feature_extractor"],
        waveforms=[np.clip(wav, -1.0, 1.0).astype(np.float32)],
        language_token_ids=[resolve_qwen_language_token_id(state["tokenizer"], language)],
        special_ids=state["special_ids"], audio_config=state["stream"],
        num_delay_tokens=[DELAY_MS // TOKEN_MS], right_pad_text_tokens=10,
        dtype=state["dtype"], device=next(model.parameters()).device,
    )[0]
    return {"text": result.get("final_text", ""), "language": language}


serve({"load": load, "transcribe": transcribe})
