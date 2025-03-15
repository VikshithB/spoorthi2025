import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import time

def pseudo_stream_whisper():
    """
    Continuously record short snippets (0.5 seconds) at 48 kHz float32,
    transcribe them using the 'medium.en' model, and print partial results.
    """
    print("Loading Whisper 'medium.en' model (English only)...")
    model = whisper.load_model("medium.en")
    print("Model loaded successfully.")

    sample_rate = 48000  # 48 kHz
    chunk_duration = 0.5 # 0.5 second snippets

    print("Starting pseudo–real-time transcription. Speak 'exit' to quit.")
    
    while True:
        # Record a short snippet
        recording = sd.rec(int(chunk_duration * sample_rate),
                           samplerate=sample_rate,
                           channels=1,
                           dtype='float32')
        sd.wait()
        
        # Save snippet to file
        temp_file = "temp_chunk.wav"
        wav.write(temp_file, sample_rate, recording)

        # Transcribe
        result = model.transcribe(temp_file, language="en")
        transcription = result["text"].strip()

        if transcription:
            print(f"[Partial] {transcription}")
        else:
            print("[Partial] (no speech detected)")

        # If user says "exit", break
        if transcription.lower() == "exit":
            print("Exiting...")
            break

        # Tiny delay to avoid spamming
        time.sleep(0.1)

if __name__ == "__main__":
    pseudo_stream_whisper()