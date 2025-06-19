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
