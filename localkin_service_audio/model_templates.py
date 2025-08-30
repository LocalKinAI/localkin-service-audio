"""
Model Templates and Helper Functions for LocalKin Service Audio

This module provides templates and utilities for easily adding new models to LocalKin Service Audio.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

# Model templates for different types
MODEL_TEMPLATES = {
    "whisper_stt": {
        "name": "whisper-medium-hf",
        "type": "stt",
        "description": "OpenAI Whisper Medium model from Hugging Face.",
        "source": "huggingface",
        "huggingface_repo": "openai/whisper-medium",
        "license": "MIT",
        "size_mb": 1500,
        "requirements": ["transformers", "torch"],
        "tags": ["medium", "balanced", "huggingface"]
    },

    "speecht5_tts": {
        "name": "speecht5-superb-hf",
        "type": "tts",
        "description": "Microsoft SpeechT5 model with SUPERB fine-tuning.",
        "source": "huggingface",
        "huggingface_repo": "microsoft/speecht5_tts",
        "license": "MIT",
        "size_mb": 1300,
        "requirements": ["transformers", "torch"],
        "tags": ["neural", "microsoft", "huggingface"]
    },

    "bark_tts": {
        "name": "bark-large-hf",
        "type": "tts",
        "description": "Suno Bark Large TTS model from Hugging Face.",
        "source": "huggingface",
        "huggingface_repo": "suno/bark",
        "license": "MIT",
        "size_mb": 5500,
        "requirements": ["transformers", "torch"],
        "tags": ["bark", "suno", "high-quality", "huggingface"]
    },

    "wav2vec2_stt": {
        "name": "wav2vec2-large-960h-hf",
        "type": "stt",
        "description": "Facebook Wav2Vec2 Large model with 960 hours of training.",
        "source": "huggingface",
        "huggingface_repo": "facebook/wav2vec2-large-960h",
        "license": "MIT",
        "size_mb": 1200,
        "requirements": ["transformers", "torch"],
        "tags": ["wav2vec2", "facebook", "large", "huggingface"]
    },

    "hubert_stt": {
        "name": "hubert-large-superb-hf",
        "type": "stt",
        "description": "Facebook HuBERT Large model with SUPERB fine-tuning.",
        "source": "huggingface",
        "huggingface_repo": "facebook/hubert-large-ls960-ft",
        "license": "MIT",
        "size_mb": 3100,
        "requirements": ["transformers", "torch"],
        "tags": ["hubert", "facebook", "superb", "huggingface"]
    },

    "fastspeech2_tts": {
        "name": "fastspeech2-en-ljspeech-hf",
        "type": "tts",
        "description": "FastSpeech2 neural TTS model trained on LJSpeech.",
        "source": "huggingface",
        "huggingface_repo": "espnet/fastspeech2_en_ljspeech",
        "license": "Apache 2.0",
        "size_mb": 800,
        "requirements": ["transformers", "torch"],
        "tags": ["fastspeech2", "espnet", "neural", "huggingface"]
    },

    "tacotron2_tts": {
        "name": "tacotron2-ljspeech-hf",
        "type": "tts",
        "description": "Tacotron2 neural TTS model trained on LJSpeech.",
        "source": "huggingface",
        "huggingface_repo": "espnet/tacotron2_ljspeech",
        "license": "Apache 2.0",
        "size_mb": 900,
        "requirements": ["transformers", "torch"],
        "tags": ["tacotron2", "espnet", "neural", "huggingface"]
    }
}

def get_model_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get a model template by name."""
    return MODEL_TEMPLATES.get(template_name)

def list_available_templates() -> List[str]:
    """List all available model templates."""
    return list(MODEL_TEMPLATES.keys())

def create_model_from_template(
    template_name: str,
    name: str,
    description: Optional[str] = None,
    repo: Optional[str] = None,
    size_mb: Optional[int] = None
) -> Dict[str, Any]:
    """Create a new model configuration from a template."""
    template = get_model_template(template_name)
    if not template:
        raise ValueError(f"Template '{template_name}' not found. Available: {list_available_templates()}")

    # Create a copy of the template
    model = template.copy()

    # Update with custom values
    model["name"] = name

    if description:
        model["description"] = description

    if repo:
        model["huggingface_repo"] = repo

    if size_mb:
        model["size_mb"] = size_mb

    return model

def validate_model_for_huggingface(model: Dict[str, Any]) -> List[str]:
    """Validate a model configuration for Hugging Face compatibility."""
    warnings = []

    # Check if repo exists and is accessible
    if "huggingface_repo" in model:
        repo = model["huggingface_repo"]
        if not repo or "/" not in repo:
            warnings.append(f"Invalid Hugging Face repo format: {repo}")
        elif len(repo.split("/")) != 2:
            warnings.append(f"Invalid Hugging Face repo format: {repo}")

    # Check requirements
    if "requirements" not in model:
        warnings.append("No requirements specified")
    elif not isinstance(model["requirements"], list):
        warnings.append("Requirements must be a list")
    elif "transformers" not in model["requirements"]:
        warnings.append("Most Hugging Face models need 'transformers' in requirements")

    return warnings

def suggest_similar_models(query: str) -> List[str]:
    """Suggest similar models based on a query."""
    query_lower = query.lower()
    suggestions = []

    for template_name, template in MODEL_TEMPLATES.items():
        # Check name similarity
        if query_lower in template_name.lower():
            suggestions.append(template_name)
        # Check description similarity
        elif query_lower in template["description"].lower():
            suggestions.append(template_name)
        # Check tags
        elif "tags" in template and any(query_lower in tag.lower() for tag in template["tags"]):
            suggestions.append(template_name)

    return suggestions[:5]  # Limit to 5 suggestions

def get_popular_models() -> List[Dict[str, Any]]:
    """Get a list of popular model configurations."""
    popular = [
        {
            "template": "whisper_stt",
            "name": "whisper-large-v3-hf",
            "repo": "openai/whisper-large-v3",
            "description": "Latest OpenAI Whisper Large v3 model"
        },
        {
            "template": "speecht5_tts",
            "name": "speecht5-tts",
            "repo": "microsoft/speecht5_tts",
            "description": "Microsoft's neural TTS model"
        },
        {
            "template": "bark_tts",
            "name": "bark-small",
            "repo": "suno/bark-small",
            "description": "Suno's Bark TTS model (smaller version)"
        },
        {
            "template": "wav2vec2_stt",
            "name": "wav2vec2-base-960h",
            "repo": "facebook/wav2vec2-base-960h",
            "description": "Facebook's Wav2Vec2 base model"
        }
    ]

    # Convert to full model configs
    models = []
    for pop in popular:
        try:
            model = create_model_from_template(
                pop["template"],
                pop["name"],
                pop["description"],
                pop["repo"]
            )
            models.append(model)
        except Exception:
            continue

    return models

def print_template_info():
    """Print information about available templates."""
    print("🎯 Available LocalKin Service Audio Model Templates:")
    print("=" * 50)

    for name, template in MODEL_TEMPLATES.items():
        print(f"\n📦 {name}")
        print(f"   Type: {template['type'].upper()}")
        print(f"   Size: {template['size_mb']}MB")
        print(f"   Description: {template['description']}")
        if "tags" in template:
            print(f"   Tags: {', '.join(template['tags'])}")

    print(f"\n💡 Use: create_model_from_template('{list(MODEL_TEMPLATES.keys())[0]}', 'my-model')")
    print("📚 Total templates available:", len(MODEL_TEMPLATES))

if __name__ == "__main__":
    print_template_info()
