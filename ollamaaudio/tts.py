import pyttsx3

def synthesize_speech(text, output_path=None):
    """
    Synthesizes speech from text using the native OS engine via pyttsx3.
    """
    try:
        engine = pyttsx3.init()
        
        if output_path:
            print(f"Saving speech to {output_path}...")
            engine.save_to_file(text, output_path)
        else:
            print("Synthesizing and playing speech...")
            engine.say(text)
            
        engine.runAndWait()
        print("Done.")
        return True

    except Exception as e:
        print(f"An unexpected error occurred during speech synthesis: {e}")
        return False
