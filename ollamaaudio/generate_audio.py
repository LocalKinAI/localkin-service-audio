import numpy as np
from scipy.io.wavfile import write

# Audio parameters
samplerate = 44100  # samples per second
frequency = 440     # Hz (A4 note)
duration = 1.0      # seconds

# Generate a sine wave
t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
amplitude = np.iinfo(np.int16).max * 0.5
data = amplitude * np.sin(2. * np.pi * frequency * t)

# Write to a .wav file
write("test_audio.wav", samplerate, data.astype(np.int16))
print("Generated test_audio.wav")
