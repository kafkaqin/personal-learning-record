import numpy as np
import matplotlib.pyplot as plt
# 时间域转换到频率域
if __name__ == '__main__':
    sampling_rate = 100
    t = np.linspace(0, 1, sampling_rate,endpoint=False) # 时间向量
    freq = 5 # 正弦波频率
    signal = np.sin(2*np.pi*freq*t)

    fft_result = np.fft.fft(signal)

    fft_freq = np.fft.fftfreq(len(signal),1/sampling_rate) # 频率轴

    n = signal.size
    single_side_fft = fft_result[:n//2]
    single_side_freq = fft_freq[:n//2]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # 绘制频谱图
    plt.subplot(1, 2, 2)
    plt.plot(single_side_freq, np.abs(single_side_fft))
    plt.title('Single-Sided Amplitude Spectrum of Sine Wave')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('SineWave.png')