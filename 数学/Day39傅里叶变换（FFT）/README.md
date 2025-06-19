使用 **NumPy 的 `np.fft.fft()`** 函数可以非常方便地对信号进行 **快速傅里叶变换（FFT）分析**，从而将时域信号转换为频域表示。

---

## 🧮 一、什么是 FFT？

**快速傅里叶变换（Fast Fourier Transform, FFT）** 是一种高效计算离散傅里叶变换（DFT）的算法。它能将一个时域信号分解成多个不同频率的正弦/余弦分量，常用于：

- 音频处理
- 图像处理
- 振动分析
- 通信系统

---

## ✅ 二、基本流程

1. **生成或加载信号**
2. **应用 `np.fft.fft()` 进行变换**
3. **计算频率轴**
4. **绘制幅度谱**

---

## 🧪 三、Python 示例代码：用 NumPy 实现 FFT 分析

```python
import numpy as np
import matplotlib.pyplot as plt

# 设置参数
fs = 1000              # 采样率（Hz）
T = 1.0 / fs           # 采样周期（秒）
L = 1000               # 信号长度（采样点数）
t = np.arange(0, L) * T  # 时间向量

# 创建一个合成信号（包含两个正弦波 + 噪声）
freq1, freq2 = 50, 120  # Hz
signal = 0.7 * np.sin(2 * np.pi * freq1 * t) + 1.5 * np.sin(2 * np.pi * freq2 * t)
signal += np.random.normal(0, 0.5, L)  # 添加噪声

# Step 1: 执行 FFT
y_fft = np.fft.fft(signal)

# Step 2: 计算频率轴（只取前一半）
frequencies = np.fft.fftfreq(L, T)

# Step 3: 取单边频谱（因为是对称的）
half_L = L // 2
fft_magnitude = np.abs(y_fft[:half_L])  # 幅度谱

# Step 4: 绘图
plt.figure(figsize=(12, 6))

# 时域信号
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("原始信号（时域）")
plt.xlabel("时间 [s]")
plt.grid()

# 频域信号
plt.subplot(2, 1, 2)
plt.plot(frequencies[:half_L], fft_magnitude)
plt.title("FFT 分析结果（频域）")
plt.xlabel("频率 [Hz]")
plt.ylabel("幅度")
plt.grid()
plt.tight_layout()
plt.show()
```

---

## 🔍 四、输出说明

- 第一张图显示了合成信号（两个正弦波 + 噪声）
- 第二张图是其对应的 **频谱图**：
    - 在 50Hz 和 120Hz 处出现明显峰值，与设定一致
    - 幅度大小也反映了信号强度

---

## 📌 五、常用函数说明

| 函数 | 含义 |
|------|------|
| `np.fft.fft(x)` | 快速傅里叶变换（复数数组） |
| `np.fft.ifft(X)` | 逆傅里叶变换 |
| `np.fft.fftfreq(n, d)` | 返回频率轴（n=样本数，d=采样间隔） |
| `np.abs(X)` | 获取幅度 |
| `np.angle(X)` | 获取相位 |

---

## 🧠 六、注意事项

- **采样定理**：要准确检测某个频率成分，采样率必须至少是该频率的两倍。
- **频谱分辨率**：分辨率为 $ \frac{f_s}{N} $，即采样率除以采样点数。
- **窗函数**：实际中建议在做 FFT 前加窗（如汉明窗），减少频谱泄漏：

```python
window = np.hamming(L)
signal_windowed = signal * window
y_fft = np.fft.fft(signal_windowed)
```

---

## 📈 七、应用场景举例

| 应用领域 | 示例 |
|----------|------|
| 音频分析 | 提取音频中的基频和泛音 |
| 故障诊断 | 通过振动信号识别设备故障频率 |
| 生物医学 | EEG、ECG 信号频谱分析 |
| 通信 | OFDM 调制解调、信道估计 |

---

