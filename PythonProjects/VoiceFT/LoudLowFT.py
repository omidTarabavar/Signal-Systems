import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the loud audio file
loud_audio, loud_sample_rate = librosa.load('Loud.mp3', sr=None)

# Load the low audio file
low_audio, low_sample_rate = librosa.load('low.mp3', sr=None)

# Compute the power of the loud signal
loud_power = np.mean(np.square(loud_audio))

# Compute the power of the low signal
low_power = np.mean(np.square(low_audio))

# Plot the loud audio signal
plt.subplot(3, 1, 1)
plt.plot(loud_audio)
plt.title("Loud Audio Signal")

# Plot the low audio signal
plt.subplot(3, 1, 2)
plt.plot(low_audio)
plt.title("Low Audio Signal")

# Plot the power values
plt.subplot(3, 1, 3)
plt.plot([0, len(loud_audio)], [loud_power, loud_power], 'r-', label=f"Loud Power: {loud_power:.4f}")
plt.plot([0, len(low_audio)], [low_power, low_power], 'g-', label=f"Low Power: {low_power:.4f}")
plt.title("Power Comparison")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()