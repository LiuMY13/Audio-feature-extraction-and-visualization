import librosa
import numpy as np
import matplotlib.pyplot as plt

# 加载音频文件，统一采样率至48kHz
file_path = "0000.wav"  # 替换为您的音频文件路径
wav, sr = librosa.load(file_path, sr=None)
print(f"sr is {sr}")
print(f"shape of wav is {wav.shape}")
# (N,)
# N = 原始音频的持续时间（秒） × 采样率（Hz）
# mono=True变成单声道

wav, sr = librosa.load(file_path, sr=48000, mono=True)
print(f"输入数据维度: {wav.shape}")

# 计算短时傅里叶变换 (STFT)
n_fft = 2048  # FFT窗口大小
hop_length = 512  # 每次移动的样本数
# 帧数
# T = （N - n_fft) / hop_size + 1
# librosa会进行padding
# T =  N / hop_size + 1

stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)

# 将幅度转换为分贝
stft_db = librosa.amplitude_to_db(np.abs(stft))
print(f"输出频谱维度: {stft_db.shape}")

# 可视化 STFT 结果
plt.figure(figsize=(15, 7))
librosa.display.specshow(
    stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="inferno"
)
plt.colorbar(format="%+2.0f dB")
plt.title("STFT Magnitude (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.savefig("STFT_vis.png")
