import whisper
import os

def transcribe_audio(model_size, audio_file_path):
    """
    Transcribes an audio file using the locally run Whisper model.
    """
    if not os.path.exists(audio_file_path):
        return f"Error: Audio file not found at {audio_file_path}"

    try:
        print(f"Loading whisper model '{model_size}'... (This might download the model on first use)")
        model = whisper.load_model(model_size)

        print(f"Transcribing {audio_file_path}...")
        result = model.transcribe(audio_file_path, fp16=False) # fp16=False for CPU

        transcribed_text = result["text"]
        print("Transcription complete.")
        return transcribed_text

    except Exception as e:
        return f"An unexpected error occurred during transcription: {e}"