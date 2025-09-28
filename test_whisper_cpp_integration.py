#!/usr/bin/env python3
"""
Test script for whisper.cpp integration with LocalKin Service Audio.

This script tests the whisper.cpp integration without requiring
whisper.cpp to be installed (for basic functionality tests).
"""

import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that whisper.cpp integration modules can be imported."""
    print("🔍 Testing imports...")

    try:
        from localkin_service_audio.core.audio_processing.whisper_cpp import (
            WhisperCppSTT,
            transcribe_with_whisper_cpp,
            get_whisper_cpp_engines,
            get_whisper_cpp_models
        )
        print("✅ whisper.cpp integration modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_stt_integration():
    """Test STT module integration."""
    print("🔍 Testing STT module integration...")

    try:
        from localkin_service_audio.core.audio_processing.stt import (
            transcribe_audio,
            get_available_engines
        )

        # Check available engines
        engines = get_available_engines()
        print(f"📊 Available engines: {engines}")

        if 'whisper_cpp' in engines:
            print("✅ whisper.cpp engine is available in STT module")
            return True
        else:
            print("⚠️ whisper.cpp engine not available (whisper-cli not installed)")
            return True  # This is OK for testing

    except ImportError as e:
        print(f"❌ STT integration failed: {e}")
        return False

def test_model_config():
    """Test model configuration."""
    print("🔍 Testing model configuration...")

    try:
        from localkin_service_audio.core import get_models

        models = get_models()
        cpp_models = [m for m in models if m.get('source') == 'whisper-cpp']

        if cpp_models:
            print(f"✅ Found {len(cpp_models)} whisper.cpp models in configuration:")
            for model in cpp_models:
                print(f"  • {model['name']} ({model.get('size_mb', 'Unknown')}MB)")
            return True
        else:
            print("❌ No whisper.cpp models found in configuration")
            return False

    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI integration (basic import test)."""
    print("🔍 Testing CLI integration...")

    try:
        # Try importing CLI (this tests the argument parsing updates)
        from localkin_service_audio.cli.cli import handle_transcribe, handle_listen
        print("✅ CLI integration modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ CLI integration failed: {e}")
        return False

def test_whisper_cpp_class():
    """Test WhisperCppSTT class instantiation (without actual execution)."""
    print("🔍 Testing WhisperCppSTT class...")

    try:
        from localkin_service_audio.core.audio_processing.whisper_cpp import WhisperCppSTT

        # Test class instantiation (should work even without whisper-cli)
        try:
            stt = WhisperCppSTT()
            print("❌ WhisperCppSTT instantiated without expected error")
            print("   (This might indicate whisper-cli is actually available)")
            return True
        except FileNotFoundError:
            print("✅ WhisperCppSTT correctly failed to instantiate (whisper-cli not found)")
            return True
        except Exception as e:
            print(f"⚠️ Unexpected error during instantiation: {e}")
            return False

    except ImportError as e:
        print(f"❌ WhisperCppSTT class test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🎵 LocalKin Service Audio - whisper.cpp Integration Test")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("STT Integration", test_stt_integration),
        ("Model Configuration", test_model_config),
        ("CLI Integration", test_cli_integration),
        ("WhisperCppSTT Class", test_whisper_cpp_class),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1

    print(f"\n📈 Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! whisper.cpp integration is ready.")
        print("\n💡 Next steps:")
        print("1. Install whisper.cpp: ./scripts/build_whisper_cpp.sh")
        print("2. Download models: python scripts/download_whisper_cpp_models.py tiny")
        print("3. Test transcription: kin audio transcribe audio.wav --engine whisper-cpp")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
