import json
import os
from typing import Dict, List, Optional, Any

MODELS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'models.json')

def get_models() -> List[Dict[str, Any]]:
    """Loads the models configuration from models.json."""
    try:
        with open(MODELS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config.get("models", [])
            # Validate each model
            validated_models = []
            for model in models:
                if validate_model_config(model):
                    validated_models.append(model)
                else:
                    print(f"Warning: Skipping invalid model configuration: {model.get('name', 'Unknown')}")
            return validated_models
    except FileNotFoundError:
        print(f"Error: The configuration file was not found at {MODELS_CONFIG_PATH}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode the JSON from {MODELS_CONFIG_PATH}: {e}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error loading configuration: {e}")
        return []

def get_config_metadata() -> Dict[str, Any]:
    """Gets the metadata from the configuration file."""
    try:
        with open(MODELS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("metadata", {})
    except Exception:
        return {}

def find_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Finds a specific model by name in the configuration."""
    models = get_models()
    for model in models:
        if model.get('name') == model_name:
            return model
    return None

def find_models_by_type(model_type: str) -> List[Dict[str, Any]]:
    """Finds all models of a specific type (stt/tts)."""
    models = get_models()
    return [model for model in models if model.get('type') == model_type]

def find_models_by_source(source: str) -> List[Dict[str, Any]]:
    """Finds all models from a specific source."""
    models = get_models()
    return [model for model in models if model.get('source') == source]

def validate_model_config(model: Dict[str, Any]) -> bool:
    """Validates a model configuration."""
    required_fields = ['name', 'type', 'description', 'source']

    # Check required fields
    for field in required_fields:
        if field not in model:
            print(f"Validation error: Missing required field '{field}' in model config")
            return False

    # Validate type
    valid_types = ['stt', 'tts']
    if model['type'] not in valid_types:
        print(f"Validation error: Invalid type '{model['type']}' for model '{model['name']}'. Must be one of: {valid_types}")
        return False

    # Validate source
    valid_sources = ['ollama', 'openai-whisper', 'pyttsx3', 'coqui-tts', 'bark-tts', 'huggingface', 'faster-whisper']
    if model['source'] not in valid_sources:
        print(f"Validation warning: Unknown source '{model['source']}' for model '{model['name']}'")

    return True

def get_model_sizes(model_name: str) -> List[str]:
    """Gets available sizes for a model."""
    model = find_model(model_name)
    if model and 'sizes' in model:
        return model['sizes']
    return []

def get_default_model_size(model_name: str) -> str:
    """Gets the default size for a model."""
    model = find_model(model_name)
    if model and 'default_size' in model:
        return model['default_size']
    return 'base'

def save_models_config(models: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Saves the models configuration to models.json."""
    try:
        config = {"models": models}
        if metadata:
            config["metadata"] = metadata
        else:
            config["metadata"] = get_config_metadata()

        with open(MODELS_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False
