#!/usr/bin/env python3
"""
Demo script showing how to use OllamaAudio API server
"""

import requests
import json
import time

def demo_api_server():
    """Demo of how the OllamaAudio API server works"""

    print("🎵 OllamaAudio API Server Demo")
    print("=" * 50)

    print("\n1. Start the server:")
    print("   ollamaaudio run whisper-tiny-hf --port 8000")

    print("\n2. Server startup output:")
    print("   🚀 Starting OllamaAudio API server for whisper-tiny-hf")
    print("   📍 Server will be available at: http://localhost:8000")
    print("   📖 API documentation: http://localhost:8000/docs")
    print("   🔧 Available API endpoints:")
    print("      GET  /           - API information")
    print("      GET  /health     - Health check")
    print("      GET  /models     - Loaded models info")
    print("      POST /transcribe - Speech to text")
    print("      GET  /docs       - Interactive API documentation")

    print("\n3. API Endpoints:")

    # Simulate API calls
    base_url = "http://localhost:8000"

    print(f"\n   📡 GET {base_url}/")
    api_info = {
        "name": "OllamaAudio API Server",
        "model": "whisper-tiny-hf",
        "type": "stt",
        "status": "running",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "GET /models": "Loaded models info",
            "POST /transcribe": "Speech to text (STT models)",
            "POST /synthesize": "Text to speech (TTS models)",
            "POST /chat": "Conversational interface (LLM models)"
        }
    }
    print(f"   Response: {json.dumps(api_info, indent=2)}")

    print(f"\n   📡 GET {base_url}/health")
    health_info = {
        "status": "healthy",
        "model": "whisper-tiny-hf",
        "loaded": True
    }
    print(f"   Response: {json.dumps(health_info, indent=2)}")

    print(f"\n   📡 GET {base_url}/models")
    models_info = {
        "loaded_models": ["whisper-tiny-hf"],
        "current_model": "whisper-tiny-hf",
        "model_info": {
            "name": "whisper-tiny-hf",
            "type": "stt",
            "source": "huggingface",
            "huggingface_repo": "openai/whisper-tiny",
            "license": "MIT",
            "size_mb": 150
        }
    }
    print(f"   Response: {json.dumps(models_info, indent=2)}")

    print("\n   📡 POST /transcribe (with audio file)")
    transcribe_example = {
        "text": "This is a sample transcription from the Whisper model.",
        "language": "en",
        "confidence": 0.95
    }
    print(f"   Response: {json.dumps(transcribe_example, indent=2)}")

    print("\n4. Interactive API Documentation:")
    print(f"   🌐 Open in browser: http://localhost:8000/docs")
    print("   📖 FastAPI automatic documentation")
    print("   🧪 Test endpoints directly in the browser")

    print("\n5. Integration Examples:")

    print("\n   🐍 Python client:")
    print("""
   import requests

   # Transcribe audio
   with open('audio.wav', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/transcribe',
           files={'file': f}
       )
   result = response.json()
   print(result['text'])
   """)

    print("\n   🌐 cURL example:")
    print("""
   curl -X POST "http://localhost:8000/transcribe" \\
        -H "accept: application/json" \\
        -H "Content-Type: multipart/form-data" \\
        -F "file=@audio.wav"
   """)

    print("\n6. Stop the server:")
    print("   🛑 Press Ctrl+C in the terminal running the server")

    print("\n✅ Demo complete! The API server provides:")
    print("   • RESTful endpoints for audio processing")
    print("   • Interactive documentation at /docs")
    print("   • Health monitoring at /health")
    print("   • Model information at /models")
    print("   • File upload support for audio files")
    print("   • JSON responses with transcription results")

if __name__ == "__main__":
    demo_api_server()
