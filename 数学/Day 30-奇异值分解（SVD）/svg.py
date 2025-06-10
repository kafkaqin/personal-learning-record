import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(path):
    img = Image.open(path).convert('L') # 灰度图
    img_matrix = np.array(img,dtype=np.float64)
    return img_matrix
def compress_image_svd(img_matrix,k):
    U,s,Vt = np.linalg.svd(img_matrix,full_matrices=False)
    compressed = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
    return compressed
def show_image(original,compressed,k):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(original,cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title(f"Compressed (k={k})")
    plt.imshow(compressed,cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("svg.png")
if __name__ == '__main__':
    image = "1749568310680.png"
    original_img = load_image(image)
    k = 30

    compressed_img = compress_image_svd(original_img,k)
    show_image(original_img,compressed_img,k)


    def compression_ratio(m, n, k):
        original_size = m * n
        compressed_size = k * (m + n + 1)  # U_k(k×m), s(k), V_k(n×k)
        ratio = original_size / compressed_size
        return ratio


    m, n = original_img.shape
    ratio = compression_ratio(m, n, k)
    print(f"压缩率（原始大小 / 压缩后大小）≈ {ratio:.2f}")