import numpy as np
import matplotlib.pyplot as plt

fs = 1000
T  = 1.0 /fs
L = 1000
t = np.arange(0, L ) * T

freq1,freq2 = 50,120
signal = 0.7 * np.sin(2 * np.pi*freq1*t)+1.5*np.sin(2 * np.pi*freq2*t)
signal +=np.random.normal(0,0.5,L)

y_fft = np.fft.fft(signal)

frequencies = np.fft.fftfreq(L,T)

half_L = L//2
fft_magnitude = np.abs(y_fft[:half_L])

plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(t,signal)
plt.title("原始信号(时域)")
plt.xlabel("时间(s)")
plt.grid()


plt.subplot(2,1,2)
plt.plot(frequencies[:half_L],fft_magnitude)
plt.title("FFT 分析结果 (频域)")
plt.xlabel("频域 [Hz]")
plt.ylabel("幅度")
plt.grid()
plt.tight_layout()
plt.savefig("fft.png")



import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: 生成测试信号
# -------------------------------

# 参数设置
fs = 1000        # 采样频率 (Hz)
T = 1.0          # 信号持续时间 (秒)
N = int(fs * T)  # 采样点数

t = np.linspace(0, T, N, endpoint=False)  # 时间轴

# 构建一个包含多个频率的信号
# 例如：50 Hz + 120 Hz 正弦波 + 噪声
f1, f2 = 50, 120
signal = (
    0.7 * np.sin(2 * np.pi * f1 * t) +
    0.5 * np.sin(2 * np.pi * f2 * t) +
    0.2 * np.random.normal(0, 1, N)  # 添加噪声
)

# -------------------------------
# Step 2: 执行 FFT
# -------------------------------

# 计算 FFT
X = np.fft.fft(signal)

# 计算对应的频率轴
freqs = np.fft.fftfreq(N, d=1/fs)  # d 是采样间隔

# 我们只关心正频率部分（前半部分）
half_N = N // 2
freqs = freqs[:half_N]
X_full = X
X = X[:half_N]

# 计算幅值谱（Magnitude Spectrum）
magnitude = np.abs(X) / half_N  # 归一化

# 或者计算功率谱
power = (np.abs(X) ** 2) / (N * N)

# -------------------------------
# Step 3: 可视化
# -------------------------------

plt.figure(figsize=(12, 6))

# 子图 1：时域信号
plt.subplot(2, 1, 1)
plt.plot(t[:200], signal[:200], 'b-', linewidth=1.2)
plt.title("时域信号（前200个点）")
plt.xlabel("时间 (s)")
plt.ylabel("幅度")
plt.grid(True, alpha=0.3)

# 子图 2：频域幅值谱
plt.subplot(2, 1, 2)
plt.plot(freqs, magnitude, 'r-', linewidth=1.2)
plt.title("频域幅值谱（FFT 结果）")
plt.xlabel("频率 (Hz)")
plt.ylabel("幅度")
plt.xlim(0, fs/2)  # 只显示奈奎斯特频率以内
plt.grid(True, alpha=0.3)

# 标出峰值频率
peak_indices = np.argsort(magnitude)[-5:]  # 找最大的5个峰值
for i in peak_indices:
    if magnitude[i] > 0.1:  # 阈值过滤
        plt.axvline(freqs[i], color='gray', linestyle='--', alpha=0.7)
        plt.text(freqs[i], magnitude[i]+0.02, f"{freqs[i]:.1f} Hz", fontsize=9)

plt.tight_layout()
plt.savefig('fft.png')


# 从频域还原时域信号
signal_recovered = np.fft.ifft(X_full).real  # X_full 是完整 FFT 结果

# 验证是否一致
print("还原误差:", np.max(np.abs(signal - signal_recovered)))