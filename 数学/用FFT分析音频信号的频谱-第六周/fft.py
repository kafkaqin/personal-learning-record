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