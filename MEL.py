import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load the audio file and set the sample rate to 48kHz
file_path = "0000.wav"  # Replace with your audio file path
wav, sr = librosa.load(file_path, sr=48000, mono=True)
print(f"Input data dimensions: {wav.shape}")

# Calculate the Mel spectrogram
n_fft = 2048  # FFT window size
hop_length = 512  # Number of samples to move each time
n_mels = 128  # Number of Mel bands
mel_spectrogram = librosa.feature.melspectrogram(
    y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
)

# Convert amplitude to decibels
mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
print(f"Output Mel spectrogram dimensions: {mel_spectrogram_db.shape}")

# Visualize the Mel spectrogram
plt.figure(figsize=(15, 7))
librosa.display.specshow(
    mel_spectrogram_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="mel",
    cmap="inferno",
)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Mel Frequency (Mel)")  # Updated label for clarity
plt.tight_layout()
# plt.show()
plt.savefig("MEL_vis.png")
