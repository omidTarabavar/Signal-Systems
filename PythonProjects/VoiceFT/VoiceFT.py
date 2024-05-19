import librosa
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


# Load the MP3 audio file
audio, sample_rate = librosa.load('voice.mp3', sr=None)

# Get the FFT of the audio signal
fft_out = fft(audio)

# Calculate the frequencies for the FFT output
frequencies = fftfreq(len(audio), 1/sample_rate)

# Shift the frequencies to have positive values first
frequencies = np.fft.fftshift(frequencies)

# Shift the FFT output to have positive values first
fft_out = np.fft.fftshift(fft_out)

# Get the magnitudes of the FFT output
magnitudes = np.abs(fft_out)

# Plot the audio signal
plt.subplot(2, 1, 1)
plt.plot(audio)
plt.title("Audio Signal")

# Plot the FFT magnitudes
plt.subplot(2, 1, 2)
plt.plot(frequencies, magnitudes)
plt.title("FFT Magnitudes")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()