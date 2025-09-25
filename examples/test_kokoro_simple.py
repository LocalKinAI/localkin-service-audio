#!/usr/bin/env python3

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Kokoro TTS...")

try:
    print("1. Importing kokoro...")
    from kokoro import KPipeline
    print("✅ Import successful")

    print("2. Creating pipeline...")
    pipeline = KPipeline(lang_code='a')
    print("✅ Pipeline created")

    print("3. Testing text generation...")
    text = "Hello world"
    voice = 'af_heart'

    print(f"   Text: {text}")
    print(f"   Voice: {voice}")

    generator = pipeline(text, voice=voice, speed=1.0)
    print("✅ Generator created")

    print("4. Processing audio segments...")
    audio_segments = []
    for i, (gs, ps, audio) in enumerate(generator):
        audio_segments.append(audio)
        print(f"   Segment {i+1}: shape {audio.shape}")

    print(f"✅ Generated {len(audio_segments)} segments")

    if audio_segments:
        print("5. Concatenating audio...")
        import numpy as np
        final_audio = np.concatenate(audio_segments)
        print(f"✅ Final audio shape: {final_audio.shape}")

        print("6. Saving to file...")
        import soundfile as sf
        output_path = "test_simple.wav"
        sf.write(output_path, final_audio, 24000)
        print(f"✅ Saved to {output_path}")

        print("🎉 SUCCESS: Kokoro TTS is working!")
    else:
        print("❌ ERROR: No audio segments generated")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
