import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

fs = 16000  # Sample rate
duration = 5  # seconds

print("Recording for 5 seconds...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished

wav.write("test_recording.wav", fs, recording)
print("Recording saved as test_recording.wav")