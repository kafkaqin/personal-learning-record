import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft,fftfreq

sample_rate, data = wavfile.read('test.wav')

if len(data.shape) >1:
    data = data[:,0]

window = np.hanning(len(data))
data = data * window

N= len(data)
yf = fft(data)
xf = fftfreq(N,1/sample_rate)

P2 = np.abs(yf/N)
P1 = P2[0:int(N/2)]
P1[1:-1] = 2*P1[1:-1]

xf = xf[0:int(N/2)]

plt.plot(xf,P1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Single-Sided Amplitude Spectrum of the Audio Single')
plt.savefig('plot.png')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs,x = wavfile.read("audio.wav")
if x.ndim > 1:
    x = x[:,0]
if x.dtype == np.int16:
    x = x.astype(np.float32)/32768.0

N = 2048
x_frame = x[0:N]
window = np.hamming(N)
x_windowed = x_frame * window
X = np.fft.fft(x_windowed)
X_half = X[0:N//2]
magnitude = np.abs(X)
power =magnitude**2
freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
plt.figure(figsize=(10, 6))
plt.plot(freqs, magnitude)
plt.xlabel('频率 (Hz)')
plt.ylabel('幅值')
plt.title('音频信号频谱')
plt.grid(True)
plt.xlim(0, fs/2)  # 只显示奈奎斯特频率以内
plt.show()

from scipy.signal import stft

f, t, Zxx = stft(x, fs, nperseg=1024)
plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)), cmap='viridis')
plt.ylabel('频率 (Hz)')
plt.xlabel('时间 (s)')
plt.colorbar(label='幅值 (dB)')
plt.title('语谱图（Spectrogram）')
plt.show()
