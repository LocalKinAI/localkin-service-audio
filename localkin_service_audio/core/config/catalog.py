"""
The model catalog: current open models, one name each, runnable anywhere.

Chosen by what people use — Hugging Face downloads, likes and trending.
Each entry lists up to two ways to run it and core/config/backends.py picks
one per machine:

- ``mlx``   mlx-audio in-process on Apple Silicon (needs the ``[mlx]`` extra)
- ``torch`` a worker in an isolated environment built on first use (CUDA,
            CPU, MPS); see core/audio_processing/isolated.py

So ``kin audio serve qwen3-asr:1.7b`` works on a Mac and on a CUDA box
alike. Models with only one backend say so in their description. MLX repos
are quantised to fit a 16 GB Mac; override ``repo_id`` in
~/.localkin-service-audio/models.json for other builds.
"""
from typing import Any, Dict, List, Optional

from ..types import HardwareRequirements, ModelConfig, ModelType


def mlx(repo: str, **parameters) -> Dict[str, Any]:
    return {"engine": "mlx-audio", "repo_id": repo, "parameters": parameters}


def torch(worker: str, repo: str, **parameters) -> Dict[str, Any]:
    return {"engine": "isolated", "repo_id": repo, "parameters": {"worker": worker, **parameters}}


_QWEN_LANG = {
    "zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese",
    "es": "spanish", "it": "italian", "auto": "auto", "default": "auto",
}
_QWEN3_TTS_LANGS = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]
# Qwen3-TTS CustomVoice speakers (ids from the model's spk_id table). Every
# one can speak all ten languages; "language" is the one it's native in.
_QWEN3_VOICE_INFO = {
    "vivian":   {"name": "Vivian · 明亮女声", "language": "zh", "gender": "female"},
    "serena":   {"name": "Serena · 温柔女声", "language": "zh", "gender": "female"},
    "uncle_fu": {"name": "Uncle Fu · 成熟男声", "language": "zh", "gender": "male"},
    "dylan":    {"name": "Dylan · 北京话男声", "language": "zh", "gender": "male"},
    "eric":     {"name": "Eric · 四川话男声", "language": "zh", "gender": "male"},
    "ryan":     {"name": "Ryan · English male", "language": "en", "gender": "male"},
    "aiden":    {"name": "Aiden · English male", "language": "en", "gender": "male"},
    "ono_anna": {"name": "Ono Anna · 日本語女声", "language": "ja", "gender": "female"},
    "sohee":    {"name": "Sohee · 한국어 여성", "language": "ko", "gender": "female"},
}
_QWEN3_TTS_VOICES = list(_QWEN3_VOICE_INFO)
# Kokoro voice-id prefix (language + gender) -> nearest Qwen3 CustomVoice
# speaker, so clients that send "zf_xiaoxiao" keep working.
_QWEN3_VOICE_ALIASES = {"zf": "vivian", "zm": "uncle_fu", "af": "serena", "bf": "serena",
                        "am": "ryan", "bm": "aiden", "jf": "ono_anna"}
_QWEN3_MLX = {"lang_arg": "lang_code", "lang_map": _QWEN_LANG}
_EU25 = ["en", "fr", "de", "es", "pt", "it", "nl", "pl", "ru", "uk", "cs", "sk", "hu",
         "ro", "bg", "hr", "sl", "da", "fi", "sv", "el", "lt", "lv", "et", "mt"]
_ASR_MULTI = ["zh", "en", "yue", "ja", "ko", "de", "fr", "es", "it", "pt", "ru", "ar"]


def _entry(name, languages, description, tags, features=(), min_ram_gb=None, **backends) -> Dict[str, Any]:
    return dict(name=name, languages=languages, description=description, tags=tags,
                features=list(features), min_ram_gb=min_ram_gb,
                backends={k: v for k, v in backends.items() if v})


STT: List[Dict[str, Any]] = [
    # --- Chinese-first -------------------------------------------------------
    _entry("qwen3-asr:0.6b", _ASR_MULTI,
           "Qwen3-ASR 0.6B - beats Whisper large-v3 on Chinese, 22 dialects, 30 languages",
           ["chinese", "multilingual", "dialects", "fast", "alibaba"], min_ram_gb=2,
           mlx=mlx("mlx-community/Qwen3-ASR-0.6B-8bit", language_names=True),
           torch=torch("transformers_asr", "Qwen/Qwen3-ASR-0.6B-hf")),
    _entry("qwen3-asr:1.7b", _ASR_MULTI,
           "Qwen3-ASR 1.7B - AISHELL-2 WER 2.71 vs 5.06 for Whisper large-v3",
           ["chinese", "multilingual", "dialects", "accurate", "alibaba"], min_ram_gb=4,
           mlx=mlx("mlx-community/Qwen3-ASR-1.7B-8bit", language_names=True),
           torch=torch("transformers_asr", "Qwen/Qwen3-ASR-1.7B-hf")),
    _entry("fireredasr2:aed", ["zh", "en"],
           "FireRedASR2 AED - lowest Mandarin CER published (AISHELL-1 0.57), 20+ dialects",
           ["chinese", "accurate", "dialects"], min_ram_gb=6,
           mlx=mlx("mlx-community/FireRedASR2-AED-mlx"),
           torch=torch("fireredasr2", "FireRedTeam/FireRedASR2-AED")),
    _entry("fun-asr:nano", ["zh", "en", "ja"],
           "Fun-ASR-Nano 800M - Tongyi's ASR, 7 Chinese dialects and 26 accents",
           ["chinese", "dialects", "alibaba"], min_ram_gb=3,
           # Not the -8bit/-4bit/-fp16 builds: they predate the Qwen3-0.6B/
           # tokenizer subfolder mlx-audio 0.5.5 loads, and fail to load.
           mlx=mlx("mlx-community/Fun-ASR-Nano-2512"),
           torch=torch("transformers_asr", "FunAudioLLM/Fun-ASR-Nano-2512-hf")),
    _entry("glm-asr:nano", ["zh", "en", "yue"],
           "GLM-ASR-Nano 1.5B - Zhipu's ASR, strong on Cantonese and quiet speech",
           ["chinese", "cantonese"], min_ram_gb=3,
           mlx=mlx("mlx-community/GLM-ASR-Nano-2512-8bit"),
           torch=torch("transformers_asr", "zai-org/GLM-ASR-Nano-2512")),
    _entry("confucius4:r2t2", ["zh", "en"],
           "Confucius4-R2T2 - NetEase Youdao's Qwen3-ASR-based recognizer (Apple Silicon only)",
           ["chinese", "education"], min_ram_gb=4,
           mlx=mlx("mlx-community/Confucius4-R2T2-8bit", language_names=True)),
    _entry("moss-transcribe-diarize", ["zh", "en"],
           "MOSS-Transcribe-Diarize 0.9B - speaker labels + timestamps (Apple Silicon only)",
           ["chinese", "diarization", "meetings"], ["diarization", "timestamps"], min_ram_gb=4,
           mlx=mlx("OpenMOSS-Team/MOSS-Transcribe-Diarize")),
    # --- Multilingual / streaming ---------------------------------------------
    _entry("whisper-mlx:large-v3-turbo", ["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
           "Whisper large-v3-turbo on MLX (elsewhere use faster-whisper:large-v3-turbo)",
           ["multilingual", "fast", "accurate"], ["timestamps"], min_ram_gb=3,
           mlx=mlx("mlx-community/whisper-large-v3-turbo-asr-fp16")),
    _entry("nemotron-asr:0.6b", ["en", "zh", "es", "de", "fr", "it", "ja", "ko", "pt", "ru", "hi", "ar", "vi", "nl"],
           "Nemotron 3.5 ASR 0.6B - NVIDIA cache-aware streaming ASR, 40 language-locales",
           ["multilingual", "streaming", "nvidia", "low-latency"], ["streaming", "timestamps"], min_ram_gb=2,
           mlx=mlx("mlx-community/nemotron-3.5-asr-streaming-0.6b"),
           torch=torch("transformers_asr", "nvidia/nemotron-3.5-asr-streaming-0.6b")),
    _entry("voxtral-realtime:4b", ["en", "zh", "fr", "es", "de", "it", "pt", "nl", "ja", "ko", "ru", "ar", "hi"],
           "Voxtral Mini 4B Realtime - Mistral's streaming ASR",
           ["multilingual", "streaming", "mistral"], ["streaming"], min_ram_gb=5,
           mlx=mlx("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"),
           torch=torch("transformers_asr", "mistralai/Voxtral-Mini-4B-Realtime-2602")),
    _entry("vibevoice-asr:9b", ["en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "ru"],
           "VibeVoice-ASR 9B - Microsoft, up to 60 min audio with speakers and hotwords",
           ["multilingual", "diarization", "long-form", "microsoft"], ["diarization", "timestamps"], min_ram_gb=8,
           mlx=mlx("mlx-community/VibeVoice-ASR-4bit", defaults={"max_tokens": 8192}),
           torch=torch("transformers_asr", "microsoft/VibeVoice-ASR-HF")),
    _entry("parakeet:0.6b", _EU25,
           "Parakeet TDT 0.6B v3 - NVIDIA, 25 European languages, far faster than Whisper",
           ["multilingual", "nvidia", "ultra-fast"], ["timestamps"], min_ram_gb=3,
           mlx=mlx("mlx-community/parakeet-tdt-0.6b-v3"),
           torch=torch("transformers_asr", "nvidia/parakeet-tdt-0.6b-v3")),
    _entry("parakeet:0.6b-v2", ["en"],
           "Parakeet TDT 0.6B v2 - NVIDIA English ASR (torch: community HF port)",
           ["english", "nvidia", "ultra-fast"], ["timestamps"], min_ram_gb=3,
           mlx=mlx("mlx-community/parakeet-tdt-0.6b-v2"),
           torch=torch("transformers_asr", "ai-and-i-project/parakeet-tdt-0.6b-v2-hf")),
    _entry("parakeet:1.1b", ["en"],
           "Parakeet TDT 1.1B - NVIDIA English ASR (torch: community HF port)",
           ["english", "nvidia"], ["timestamps"], min_ram_gb=5,
           mlx=mlx("mlx-community/parakeet-tdt-1.1b"),
           torch=torch("transformers_asr", "extraordinarylab/parakeet-tdt-1.1b")),
    _entry("canary:1b-v2", _EU25,
           "Canary 1B v2 - NVIDIA transcription + translation, 25 languages (no auto-detect)",
           ["multilingual", "nvidia", "translation"], ["translation"], min_ram_gb=6,
           mlx=mlx("nvidia/canary-1b-v2"),
           torch=torch("transformers_asr", "nvidia/canary-1b-v2")),
    _entry("canary-qwen:2.5b", ["en"],
           "Canary-Qwen 2.5B - NVIDIA English ASR leaderboard leader (Apple Silicon only)",
           ["english", "nvidia", "accurate"], min_ram_gb=8,
           mlx=mlx("nvidia/canary-qwen-2.5b")),
]

TTS: List[Dict[str, Any]] = [
    # --- Chinese-first -------------------------------------------------------
    _entry("qwen3-tts:0.6b", _QWEN3_TTS_LANGS,
           "Qwen3-TTS 0.6B CustomVoice - 9 preset voices incl. dialects, emotion via instruct",
           ["chinese", "multilingual", "fast", "alibaba"], ["emotion", "streaming", "cross_lingual"], min_ram_gb=3,
           mlx=mlx("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", defaults={"voice": "vivian"},
                   voices=_QWEN3_TTS_VOICES, voice_info=_QWEN3_VOICE_INFO, voice_aliases=_QWEN3_VOICE_ALIASES, **_QWEN3_MLX),
           torch=torch("qwen3_tts", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", defaults={"voice": "Vivian"},
                       voices=_QWEN3_TTS_VOICES, voice_info=_QWEN3_VOICE_INFO, voice_aliases=_QWEN3_VOICE_ALIASES)),
    _entry("qwen3-tts:1.7b", _QWEN3_TTS_LANGS,
           "Qwen3-TTS 1.7B CustomVoice - most-downloaded TTS of 2026, emotion via instruct",
           ["chinese", "multilingual", "quality", "alibaba"], ["emotion", "streaming", "cross_lingual"], min_ram_gb=5,
           mlx=mlx("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", defaults={"voice": "vivian"},
                   voices=_QWEN3_TTS_VOICES, voice_info=_QWEN3_VOICE_INFO, voice_aliases=_QWEN3_VOICE_ALIASES, **_QWEN3_MLX),
           torch=torch("qwen3_tts", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", defaults={"voice": "Vivian"},
                       voices=_QWEN3_TTS_VOICES, voice_info=_QWEN3_VOICE_INFO, voice_aliases=_QWEN3_VOICE_ALIASES)),
    _entry("qwen3-tts:1.7b-voicedesign", _QWEN3_TTS_LANGS,
           "Qwen3-TTS 1.7B VoiceDesign - describe the voice in words (pass instruct)",
           ["chinese", "multilingual", "voice-design", "alibaba"], ["voice_design"], min_ram_gb=5,
           mlx=mlx("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit", **_QWEN3_MLX),
           torch=torch("qwen3_tts", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")),
    _entry("qwen3-tts:0.6b-base", _QWEN3_TTS_LANGS,
           "Qwen3-TTS 0.6B Base - 3-second zero-shot voice cloning",
           ["chinese", "multilingual", "voice-cloning", "alibaba"], ["voice_cloning"], min_ram_gb=3,
           mlx=mlx("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit", needs_reference=True, **_QWEN3_MLX),
           torch=torch("qwen3_tts", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", needs_reference=True)),
    _entry("qwen3-tts:1.7b-base", _QWEN3_TTS_LANGS,
           "Qwen3-TTS 1.7B Base - higher-fidelity zero-shot voice cloning",
           ["chinese", "multilingual", "voice-cloning", "alibaba"], ["voice_cloning"], min_ram_gb=5,
           mlx=mlx("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit", needs_reference=True, **_QWEN3_MLX),
           torch=torch("qwen3_tts", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", needs_reference=True)),
    _entry("voxcpm2", ["zh", "en", "ja", "ko", "de", "fr", "es", "it", "pt", "ru", "yue"],
           "VoxCPM2 2B - Seed-TTS zh CER 0.97, 48 kHz, voice design and 9 Chinese dialects",
           ["chinese", "multilingual", "voice-cloning", "quality", "openbmb"],
           ["voice_cloning", "voice_design", "streaming"], min_ram_gb=5,
           mlx=mlx("mlx-community/VoxCPM2-8bit"),
           torch=torch("voxcpm", "openbmb/VoxCPM2")),
    _entry("cosyvoice3:0.5b", ["zh", "en", "ja", "ko", "yue", "de", "fr", "ru", "es", "it"],
           "Fun-CosyVoice3 0.5B - Tongyi, zh CER 1.12, 18+ dialects, instruct control (CUDA/CPU)",
           ["chinese", "multilingual", "voice-cloning", "dialects", "alibaba"],
           ["voice_cloning", "cross_lingual", "emotion"], min_ram_gb=4,
           torch=torch("cosyvoice", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", needs_reference=True)),
    _entry("cosyvoice2:0.5b", ["zh", "en", "ja", "ko", "yue", "de", "fr", "ru", "es", "it"],
           "CosyVoice2 0.5B - 9 languages + 18 Chinese dialects, voice cloning (CUDA/CPU)",
           ["chinese", "voice-cloning", "streaming", "alibaba"], ["voice_cloning", "cross_lingual"],
           min_ram_gb=4,
           torch=torch("cosyvoice", "FunAudioLLM/CosyVoice2-0.5B", needs_reference=True)),
    _entry("cosyvoice:300m", ["zh", "en", "ja", "ko", "yue"],
           "CosyVoice 300M SFT - original CosyVoice with 7 preset voices (CUDA/CPU)",
           ["chinese", "alibaba"], [], min_ram_gb=3,
           torch=torch("cosyvoice", "FunAudioLLM/CosyVoice-300M-SFT", defaults={"voice": "中文女"},
                       voices=["中文女", "中文男", "日语男", "粤语女", "英文女", "英文男", "韩语女"])),
    # No mlx backend: the mlx-community IndexTTS builds don't load in
    # mlx-audio 0.5.5 (ModelArgs lacks bigvgan/tokenizer_name).
    _entry("indextts:2", ["zh", "en"],
           "IndexTTS-2 - bilibili's zero-shot TTS with emotion control",
           ["chinese", "voice-cloning", "emotional", "bilibili"], ["voice_cloning", "emotion"], min_ram_gb=6,
           torch=torch("indextts", "IndexTeam/IndexTTS-2", needs_reference=True)),
    _entry("indextts:2.5", ["zh", "en"],
           "IndexTTS-2.5 - newer IndexTTS, faster and more stable",
           ["chinese", "voice-cloning", "emotional", "bilibili"], ["voice_cloning", "emotion"], min_ram_gb=6,
           torch=torch("indextts", "IndexTeam/IndexTTS-2.5", needs_reference=True)),
    _entry("fish-speech:s2-pro", ["zh", "en", "ja", "ko", "de", "fr", "es"],
           "Fish Audio S2 Pro - inline emotion tags, voice cloning (Apple Silicon only)",
           ["chinese", "multilingual", "voice-cloning", "quality"], ["voice_cloning", "emotion"], min_ram_gb=8,
           mlx=mlx("mlx-community/fish-audio-s2-pro-8bit")),
    _entry("spark-tts:0.5b", ["zh", "en"],
           "Spark-TTS 0.5B - gender and pitch control, voice cloning (Apple Silicon only)",
           ["chinese", "voice-cloning"], ["voice_cloning"], min_ram_gb=3,
           mlx=mlx("mlx-community/Spark-TTS-0.5B-bf16")),
    _entry("longcat-audiodit:1b", ["zh", "en"],
           "LongCat-AudioDiT 1B - Meituan diffusion TTS, zh CER 1.18 (Apple Silicon only)",
           ["chinese", "voice-cloning", "quality"], ["voice_cloning"], min_ram_gb=7,
           mlx=mlx("mlx-community/LongCat-AudioDiT-1B-bf16")),
    _entry("breeze-tts:2", ["zh", "en"],
           "Breeze-TTS-2 3.5B - Taiwanese Mandarin + English, voice design (Apple Silicon only)",
           ["chinese", "taiwanese", "voice-cloning"], ["voice_cloning", "voice_design"], min_ram_gb=6,
           mlx=mlx("mlx-community/Breeze-TTS-2-mlx-8bit")),
    _entry("confucius4-tts", ["zh", "en", "ja", "ko", "de", "fr", "es", "pt", "it", "ru", "vi", "th", "id", "ms"],
           "Confucius4-TTS - NetEase Youdao, 14 languages (Apple Silicon only)",
           ["chinese", "multilingual", "voice-cloning"], ["voice_cloning"], min_ram_gb=4,
           mlx=mlx("mlx-community/Confucius4-TTS-mlx-int8")),
    _entry("moss-tts:nano", ["zh", "en", "ja", "ko", "de", "fr", "es"],
           "MOSS-TTS-Nano 100M - tiny multilingual cloning TTS (Apple Silicon only)",
           ["chinese", "multilingual", "lightweight", "voice-cloning"], ["voice_cloning", "streaming"],
           min_ram_gb=1, mlx=mlx("mlx-community/MOSS-TTS-Nano-100M", needs_reference=True)),
    _entry("moss-tts:local-v1.5", ["zh", "en", "ja", "ko", "de", "fr", "es"],
           "MOSS-TTS Local Transformer v1.5 - 31 languages (Apple Silicon only)",
           ["chinese", "multilingual", "voice-cloning"], ["voice_cloning"], min_ram_gb=7,
           mlx=mlx("mlx-community/MOSS-TTS-Local-Transformer-v1.5-8bit", needs_reference=True)),
    _entry("ming-omni-tts:0.5b", ["zh", "en"],
           "Ming-Omni-TTS 0.5B - Ant Group, style control (Apple Silicon only)",
           ["chinese", "voice-cloning"], ["voice_cloning", "emotion"], min_ram_gb=2,
           mlx=mlx("mlx-community/Ming-omni-tts-0.5B-4bit")),
    # --- Multilingual --------------------------------------------------------
    _entry("chatterbox:multilingual",
           ["en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "ru", "ar", "hi", "nl", "pl", "tr"],
           "Chatterbox Multilingual - Resemble AI, 23 languages, emotion exaggeration control",
           ["multilingual", "voice-cloning", "emotional"], ["voice_cloning", "emotion"], min_ram_gb=5,
           # The MLX build ships no default voice conditionals.
           mlx=mlx("mlx-community/chatterbox-multilingual-v3", lang_arg="lang_code", needs_reference=True,
                   lang_map={"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "default": "en"}),
           torch=torch("chatterbox", "ResembleAI/chatterbox")),
    _entry("chatterbox:turbo", ["en"],
           "Chatterbox Turbo - Resemble AI's fast English TTS with paralinguistic tags",
           ["english", "fast", "voice-cloning"], ["voice_cloning", "emotion"], min_ram_gb=5,
           mlx=mlx("mlx-community/chatterbox-turbo-fp16"),
           torch=torch("chatterbox", "ResembleAI/chatterbox-turbo")),
    _entry("omnivoice", ["zh", "en", "ja", "ko", "de", "fr", "es", "ru"],
           "OmniVoice 0.6B - k2-fsa zero-shot TTS for 600+ languages, nonverbal tags",
           ["multilingual", "voice-cloning"], ["voice_cloning"], min_ram_gb=3,
           mlx=mlx("mlx-community/OmniVoice-bf16", lang_arg="language", lang_map=_QWEN_LANG),
           torch=torch("omnivoice", "k2-fsa/OmniVoice", needs_reference=True)),
    _entry("higgs-audio:v2", ["en", "zh", "ko", "de", "es"],
           "Higgs Audio v2 3B - Boson AI expressive TTS, multi-speaker dialogue",
           ["multilingual", "expressive", "voice-cloning"], ["voice_cloning", "emotion"], min_ram_gb=8,
           mlx=mlx("mlx-community/higgs-audio-v2-3B-mlx-q8"),
           torch=torch("transformers_tts", "bosonai/higgs-tts-2-3b-base", needs_reference=True)),
    _entry("outetts:1b", ["en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "ru", "ar", "nl"],
           "OuteTTS 1.0 1B - Llama-based multilingual TTS (Apple Silicon only)",
           ["multilingual", "voice-cloning"], ["voice_cloning"], min_ram_gb=3,
           mlx=mlx("mlx-community/Llama-OuteTTS-1.0-1B-8bit")),
    _entry("voxtral-tts:4b", ["en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi"],
           "Voxtral 4B TTS - Mistral, 20 preset voices, no Chinese (Apple Silicon only)",
           ["multilingual", "mistral"], [], min_ram_gb=5,
           mlx=mlx("mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", defaults={"voice": "neutral_female"})),
    # --- English -------------------------------------------------------------
    _entry("vibevoice:0.5b", ["en"],
           "VibeVoice Realtime 0.5B - Microsoft streaming TTS (Apple Silicon only)",
           ["english", "streaming", "microsoft"], ["streaming"], min_ram_gb=3,
           mlx=mlx("mlx-community/VibeVoice-Realtime-0.5B-8bit")),
    _entry("vibevoice:1.5b", ["en", "zh"],
           "VibeVoice 1.5B - Microsoft long-form multi-speaker TTS",
           ["english", "multi-speaker", "microsoft"], ["voice_cloning"], min_ram_gb=6,
           torch=torch("transformers_tts", "vibevoice/VibeVoice-1.5B-hf", needs_reference=True)),
    _entry("csm:1b", ["en"],
           "Sesame CSM 1B - conversational speech model with voice cloning",
           ["english", "conversational", "voice-cloning"], ["voice_cloning", "conversational"], min_ram_gb=3,
           mlx=mlx("mlx-community/csm-1b-8bit"),
           torch=torch("transformers_tts", "eustlb/csm-1b", needs_reference=True)),
    _entry("dia:1.6b", ["en"],
           "Dia 1.6B - multi-speaker dialogue in one pass ([S1]/[S2]), laughter and coughs",
           ["english", "dialogue", "multi-speaker", "expressive"], ["dialogue", "multi_speaker", "nonverbal"],
           min_ram_gb=5,
           mlx=mlx("mlx-community/Dia-1.6B-fp16"),
           torch=torch("transformers_tts", "nari-labs/Dia-1.6B-0626", needs_reference=True)),
    _entry("orpheus:3b", ["en"],
           "Orpheus 3B - Llama-based emotional TTS, <laugh> <sigh> tags (Apple Silicon only)",
           ["english", "emotional", "expressive"], ["emotion"], min_ram_gb=6,
           mlx=mlx("mlx-community/orpheus-3b-0.1-ft-8bit", defaults={"voice": "tara"},
                   voices=["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"])),
    _entry("kitten-tts:mini", ["en"], "KittenTTS mini 0.8 - under 100 MB (Apple Silicon only)",
           ["english", "lightweight"], min_ram_gb=1, mlx=mlx("mlx-community/kitten-tts-mini-0.8")),
    _entry("soprano:80m", ["en"], "Soprano 1.1 80M - tiny, very fast English TTS (Apple Silicon only)",
           ["english", "lightweight", "fast"], ["streaming"], min_ram_gb=1,
           mlx=mlx("mlx-community/Soprano-1.1-80M-bf16")),
    _entry("pocket-tts", ["en", "fr", "de", "pt", "it", "es"],
           "Kyutai Pocket TTS - 100M, real-time on CPU, voice cloning (Apple Silicon only)",
           ["multilingual", "lightweight", "voice-cloning"], ["voice_cloning"], min_ram_gb=1,
           mlx=mlx("mlx-community/pocket-tts")),
    _entry("irodori-tts", ["ja"], "Irodori-TTS v4.1 Small - Japanese TTS (Apple Silicon only)",
           ["japanese", "voice-cloning"], ["voice_cloning"], min_ram_gb=3,
           mlx=mlx("mlx-community/Irodori-TTS-v4.1-Small-8bit")),
]


def _config(model_type: ModelType, entry: Dict[str, Any]) -> ModelConfig:
    backends = entry["backends"]
    first = next(iter(backends.values()))
    notes = " / ".join(
        {"mlx": "Apple Silicon (mlx-audio)", "torch": "CUDA/CPU/MPS (isolated env)"}[b] for b in backends
    )
    return ModelConfig(
        name=entry["name"],
        type=model_type,
        # Replaced by the chosen backend's engine in resolve_backend().
        engine=first["engine"],
        repo_id=first["repo_id"],
        languages=list(entry["languages"]),
        features=list(entry["features"]),
        description=entry["description"],
        tags=list(entry["tags"]) + [f"backend:{b}" for b in backends],
        hardware_requirements=HardwareRequirements(min_ram_gb=entry["min_ram_gb"], performance_notes=notes),
        backends=backends,
    )


def catalog_models() -> Dict[str, ModelConfig]:
    models = {e["name"]: _config(ModelType.STT, e) for e in STT}
    models.update({e["name"]: _config(ModelType.TTS, e) for e in TTS})
    return models
