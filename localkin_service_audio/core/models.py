import requests
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib

OLLAMA_API_URL = "http://localhost:11434"

# Cache configuration
CACHE_DIR = Path.home() / ".localkin-service-audio" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR = CACHE_DIR / "huggingface"
HF_CACHE_DIR.mkdir(exist_ok=True)

def list_local_models():
    """Gets the list of models available locally from the Ollama API."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        response.raise_for_status()
        models_data = response.json()
        return [model['name'] for model in models_data.get('models', [])]
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        return None

def pull_model(model_name, source="ollama", huggingface_repo=None):
    """Pulls a model from various sources (Ollama, Hugging Face)."""
    if source == "ollama":
        return pull_ollama_model(model_name)
    elif source == "huggingface":
        return pull_huggingface_model(model_name, huggingface_repo)
    else:
        print(f"❌ Unsupported source: {source}")
        return False

def pull_ollama_model(model_name):
    """Pulls a model from the Ollama registry."""
    try:
        print(f"📥 Pulling Ollama model: {model_name}. This may take a while...")
        with requests.post(
            f"{OLLAMA_API_URL}/api/pull",
            json={"name": model_name},
            stream=True
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    line = json.loads(chunk)
                    if "status" in line:
                        status_msg = line["status"]
                        if "pulling" in status_msg.lower():
                            print(f"📥 {status_msg}", end='\r', flush=True)
                        elif "verifying" in status_msg.lower():
                            print(f"✅ {status_msg}", end='\r', flush=True)
                        else:
                            print(f"⏳ {status_msg}", end='\r', flush=True)
            print("\n✅ Ollama model pulled successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error pulling model from Ollama API: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing response from Ollama API: {e}")
        return False

def pull_huggingface_model(model_name, repo_id):
    """Pulls a model from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"📥 Downloading model from Hugging Face: {repo_id}")

        # Create cache directory for this model
        model_cache_dir = HF_CACHE_DIR / model_name
        model_cache_dir.mkdir(exist_ok=True)

        # Download the model
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=model_cache_dir,
            local_dir=model_cache_dir
        )

        print(f"✅ Model downloaded to: {local_path}")
        return True

    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        return False

def run_ollama_model(model_name, port=8000):
    """Runs an Ollama model as a server."""
    try:
        print(f"🚀 Starting Ollama model server: {model_name} on port {port}")

        # Check if Ollama is running
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
            response.raise_for_status()
        except:
            print("❌ Ollama is not running. Please start Ollama first:")
            print("   brew services start ollama  # macOS")
            print("   ollama serve                 # Linux/Windows")
            return False

        # Check if model exists locally
        local_models = list_local_models()
        if local_models is None:
            print("❌ Cannot connect to Ollama")
            return False

        if model_name not in local_models:
            print(f"❌ Model '{model_name}' not found locally. Pull it first:")
            print(f"   kin audio pull {model_name}")
            return False

        print(f"✅ Model '{model_name}' is ready!")
        print(f"🌐 Ollama API is available at: {OLLAMA_API_URL}")
        print(f"📖 Use this model with: kin audio run {model_name}")
        print("💡 Or integrate with your applications using the Ollama API")
        return True

    except Exception as e:
        print(f"❌ Error running model: {e}")
        return False

def run_huggingface_model(model_name: str, port: int = 8000):
    """Runs a Hugging Face model as an API server."""
    try:
        from ..api.server import run_server
        from .config import find_model

        model_info = find_model(model_name)
        if not model_info or model_info.get("source") != "huggingface":
            print(f"❌ Model '{model_name}' not found or not a Hugging Face model")
            return False

        # Check if model is cached
        cache_info = get_cache_info()
        cached_models = [m["name"] for m in cache_info["cached_models"]]
        if model_name not in cached_models:
            print(f"📥 Model '{model_name}' not found in cache. Pulling it first...")
            success = pull_model(model_name, "huggingface", model_info.get("huggingface_repo"))
            if not success:
                print(f"❌ Failed to pull model '{model_name}'")
                return False
            print("✅ Model pulled successfully!")

        print(f"🚀 Starting LocalKin Service Audio API server for {model_name}")
        print(f"📍 Server will be available at: http://localhost:{port}")
        print(f"📖 API documentation: http://localhost:{port}/docs")
        print("🛑 Press Ctrl+C to stop the server")
        # This will block until the server is stopped
        run_server(model_name, host="0.0.0.0", port=port)
        return True

    except Exception as e:
        print(f"❌ Error running model: {e}")
        return False

def get_cache_info():
    """Gets information about cached models."""
    cache_info = {
        "cache_dir": str(CACHE_DIR),
        "huggingface_cache": str(HF_CACHE_DIR),
        "cached_models": []
    }

    # Check Hugging Face cache
    if HF_CACHE_DIR.exists():
        for model_dir in HF_CACHE_DIR.iterdir():
            if model_dir.is_dir():
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                cache_info["cached_models"].append({
                    "name": model_dir.name,
                    "source": "huggingface",
                    "path": str(model_dir),
                    "size_mb": round(size / (1024 * 1024), 2)
                })

    return cache_info

def clear_cache(model_name=None):
    """Clears the cache for a specific model or all models."""
    if model_name:
        model_cache = HF_CACHE_DIR / model_name
        if model_cache.exists():
            import shutil
            shutil.rmtree(model_cache)
            print(f"🗑️  Cleared cache for model: {model_name}")
            return True
        else:
            print(f"❌ Model '{model_name}' not found in cache")
            return False
    else:
        if CACHE_DIR.exists():
            import shutil
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            HF_CACHE_DIR.mkdir(exist_ok=True)
            print("🗑️  Cleared all cached models")
            return True
        return False
