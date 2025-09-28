#!/usr/bin/env python3
"""
Test script for streaming LLM functionality in LocalKin Service Audio.

This script demonstrates how the streaming feature works with Ollama.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localkin_service_audio.cli.cli import query_ollama_llm_streaming

def test_streaming():
    """Test the streaming LLM functionality."""
    print("🧪 Testing streaming LLM functionality...")
    print("Note: This requires Ollama to be running locally on port 11434")
    print()

    # Test streaming generator
    try:
        print("🤖 Testing streaming response generator...")
        response_gen = query_ollama_llm_streaming("Hello, how are you?", "qwen3:14b")

        print("Streaming response: ", end="", flush=True)
        token_count = 0
        for token, full_response in response_gen:
            print(token, end="", flush=True)
            token_count += 1
            if token_count > 50:  # Limit for demo
                print("\n[...truncated for demo...]")
                break
        print()

    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        print("This is expected if Ollama is not running.")
        return False

    print("✅ Streaming functionality test completed!")
    return True

if __name__ == "__main__":
    test_streaming()
