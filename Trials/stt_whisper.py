import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper

# Configuration parameters
FS = 16000         # Sample rate (Hz)
DURATION = 10      # Recording duration (seconds)
AUDIO_FILENAME = "test_recording.wav"

def record_audio(duration, fs):
    """Record audio from the default microphone."""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    return np.squeeze(recording)

def save_audio(filename, fs, audio_data):
    """Save the recorded audio to a WAV file."""
    wav.write(filename, fs, audio_data)
    print(f"Audio saved as {filename}")

def transcribe_and_translate(audio_file):
    """Load the medium Whisper model and transcribe (and translate) the audio into English."""
    print("Loading Whisper model (medium)...")
    model = whisper.load_model("medium")
    print("Transcribing and translating audio to English...")
    # The 'task' parameter is set to 'translate' to translate any input language into English.
    result = model.transcribe(audio_file, task='translate')
    return result["text"]

if __name__ == "__main__":
    # Record audio
    audio_data = record_audio(DURATION, FS)
    # Save the audio to a file for processing
    save_audio(AUDIO_FILENAME, FS, audio_data)
    # Transcribe and translate the recorded audio
    transcription = transcribe_and_translate(AUDIO_FILENAME)
    print("Transcription:", transcription)