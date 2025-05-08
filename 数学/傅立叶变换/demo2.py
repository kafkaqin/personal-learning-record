import numpy as np
import matplotlib.pyplot as plt
# 时间域转换到频率域
if __name__ == '__main__':
    # 生成一个二维信号（例如一张图像）
    image = np.random.rand(128, 128)

    # 进行二维傅立叶变换
    fft_image = np.fft.fft2(image)
    fft_image_shifted = np.fft.fftshift(fft_image)  # 将零频率分量移到中心

    # 绘制频谱图
    magnitude_spectrum = np.log(np.abs(fft_image_shifted))  # 对数尺度显示

    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.savefig('MagnitudeSpectrum.png')