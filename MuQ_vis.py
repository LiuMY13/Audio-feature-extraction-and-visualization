import torch
import librosa
import matplotlib.pyplot as plt
from muq import MuQ

# ----------------------------
# 配置
# ----------------------------
device = "cuda"  # 使用 CUDA
file_path = "/calc/users/cisri_shzh_gpu/users/lmy/lyra/Audio-feature-extraction-and-visualization/0000.wav"
model_name = "OpenMuQ/MuQ-large-msd-iter"
selected_layers = [0, 6, 12]  # 要可视化的层

# ----------------------------
# 加载音频
# ----------------------------
print("Loading audio...")
wav, sr = librosa.load(file_path, sr=24000)
wavs = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
print(f"Audio shape: {wavs.shape}, sample rate: {sr} Hz")

# ----------------------------
# 加载模型
# ----------------------------
print("Loading MuQ model...")
muq = MuQ.from_pretrained(model_name)
muq = muq.to(device).eval()

# ----------------------------
# 推理
# ----------------------------
print("Running inference...")
with torch.no_grad():
    output = muq(wavs, output_hidden_states=True)

print(f"Total number of layers: {len(output.hidden_states)}")
print(
    f"Last hidden state shape: {output.last_hidden_state.shape}"
)  # (batch, time, hidden_dim)

# ----------------------------
# 可视化指定层
# ----------------------------
print("Visualizing hidden states...")

# 获取实际时间帧数和隐藏维度
sample_hidden = output.hidden_states[0].squeeze(0).cpu().numpy()
num_frames, hidden_dim = sample_hidden.shape
print(f"Feature map shape per layer: ({num_frames}, {hidden_dim})")

# 创建子图：1 行，3 列
fig, axes = plt.subplots(1, len(selected_layers), figsize=(18, 5))

for idx, layer in enumerate(selected_layers):
    # 获取隐藏状态并转为 NumPy
    hidden = output.hidden_states[layer].squeeze(0).cpu().numpy()  # (T, D)

    # 转置以便时间在 x 轴，特征维度在 y 轴
    ax = axes[idx]
    im = ax.imshow(
        hidden.T,  # 转置：(D, T) → y=dim, x=time
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",  # 红-蓝对称色图，适合正负激活
        vmin=-3,
        vmax=3,  # 限制颜色范围以增强对比
    )
    ax.set_title(f"Layer {layer}", fontsize=14)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Hidden Dimension")

    # 设置 tick（自动或简化）
    ax.set_xticks([0, num_frames // 2, num_frames - 1])
    ax.set_xticklabels([0, num_frames // 2, num_frames - 1])
    ax.set_yticks([0, hidden_dim // 2, hidden_dim - 1])
    ax.set_yticklabels([0, hidden_dim // 2, hidden_dim - 1])

    # 添加 colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Activation")

plt.tight_layout()
output_path = "muq_hidden_states.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Visualization saved to: {output_path}")
