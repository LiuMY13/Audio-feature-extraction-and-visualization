import torch
import json
import numpy as np
import librosa
import soundfile as sf
from stable_audio_tools.models.factory import create_model_from_config

# 加载模型配置
model_path = "/calc/users/cisri_shzh_gpu/users/lmy/lyra/stable-audio-tools/stability"
config_path = f"{model_path}/model_config.json"

with open(config_path, "r") as f:
    model_config = json.load(f)

# 完全禁用 T5 文本编码器（避免网络请求）
if "model" in model_config and "conditioning" in model_config["model"]:
    cond_config = model_config["model"]["conditioning"]
    if "configs" in cond_config:
        # 过滤掉 T5 相关的配置，只保留非文本的 conditioning
        filtered_configs = []
        for config in cond_config["configs"]:
            if config.get("type") != "t5":  # 移除 T5 配置
                filtered_configs.append(config)
        cond_config["configs"] = filtered_configs
        print("Filtered out T5 conditioning configs for offline use.")

        # 如果没有其他条件器了，可能需要完全移除 conditioning
        if len(filtered_configs) == 0:
            del model_config["model"]["conditioning"]
            print("Removed conditioning entirely as no non-T5 conditioners remain.")

# 创建模型
model = create_model_from_config(model_config)

# 加载权重
from safetensors.torch import load_file

state_dict = load_file(f"{model_path}/model.safetensors")
model.load_state_dict(state_dict, strict=False)
print("Model weights loaded successfully!")

# 提取 VAE
vae = model.pretransform
vae.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = vae.to(device)


# 音频预处理
def load_and_preprocess_audio(audio_path, target_sr):
    # 加载音频 - 使用 librosa
    data, sr = librosa.load(audio_path, sr=None, mono=False)
    print(f"Original audio shape: {data.shape}, sample rate: {sr}")

    # 确保是二维数组 (channels, samples)
    data = np.asarray(data)
    if data.ndim == 1:
        data = np.expand_dims(data, 0)

    audio = torch.from_numpy(data).float()

    # 确保双声道
    if audio.shape[0] == 1:  # 单声道 → 双声道
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:  # 多声道 → 双声道
        audio = audio[:2, :]

    # 重采样 - 使用 librosa
    if sr != target_sr:
        data = audio.numpy()
        resampled = [
            librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in data
        ]
        audio = torch.from_numpy(np.stack(resampled, axis=0)).float()

    # 归一化
    audio = audio / audio.abs().max().clamp(min=1e-5)

    return audio.unsqueeze(0)  # [1, 2, T]


# 执行重建
audio_path = "/calc/users/cisri_shzh_gpu/users/lmy/lyra/Audio-feature-extraction-and-visualization/0000.wav"
sample_rate = model_config["sample_rate"]

audio = load_and_preprocess_audio(audio_path, sample_rate).to(device)
print(f"Input audio shape: {audio.shape}")

with torch.no_grad():
    latent = vae.encode(audio)
    print(f"Latent shape: {latent.shape}")
    reconstructed = vae.decode(latent)
    print(f"Reconstructed shape: {reconstructed.shape}")

# 保存结果
reconstructed = reconstructed.squeeze(0).cpu()  # [B,C,T] -> [C,T]
reconstructed = reconstructed[:1, :]  # 只保留第一个通道
print(f"Reconstructed shape: {reconstructed.shape}")
sf.write("reconstructed_final_vae_fixed.wav", reconstructed.numpy().T, sample_rate)
print("Final reconstruction saved!")
