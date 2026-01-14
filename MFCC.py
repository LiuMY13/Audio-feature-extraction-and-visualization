import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load the audio file and set the sample rate to 48kHz
file_path = "0000.wav"  # Replace with your audio file path
wav, sr = librosa.load(file_path, sr=48000, mono=True)
print(f"Input data dimensions: {wav.shape}")

# Compute MFCCs
n_fft = 2048  # FFT window size
hop_length = 512  # Number of samples to shift each frame
n_mels = 128  # Number of Mel filters (used internally)
n_mfcc = 13  # Number of MFCC coefficients to keep

mfcc = librosa.feature.mfcc(
    y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc
)

print(f"Output MFCC dimensions: {mfcc.shape}")  # e.g., (13, 539)

# Visualize MFCC
plt.figure(figsize=(15, 7))
librosa.display.specshow(
    mfcc, sr=sr, hop_length=hop_length, x_axis="time", cmap="viridis"
)
plt.colorbar(format="%+2.0f")
plt.title("MFCC (Mel-Frequency Cepstral Coefficients)")
plt.xlabel("Time (s)")
plt.ylabel("MFCC Coefficient")
plt.tight_layout()
plt.savefig("MFCC_vis.png", dpi=300, bbox_inches="tight")
# plt.show()  # Uncomment if you want to display
