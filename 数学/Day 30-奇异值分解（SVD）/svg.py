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



    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import linalg

    img = np.random.rand(100,100) * 255

    img = img[:150,:150]
    print(f"图像的形状{img.shape}")

    U,sigma,Vt=linalg.svd(img, full_matrices=False)
    print(f"奇异值数量:{len(sigma)}")
    print(f"前5个奇异值:{sigma[:5]}")

    def compress_image(U,sigma,Vt,k):
        U_k = U[:,:k]
        sigma_k = sigma[:k]
        Vt_k = Vt[:k,:]
        image_compressed = U_k @ np.diag(sigma_k) @ Vt_k
        return image_compressed

    k_values = [10,20,50,100]
    plt.figure(figsize=(12, 8))

    # 原图
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'原图 (尺寸: {img.shape})')
    plt.axis('off')
    # 不同 k 的压缩图
    for i, k in enumerate(k_values):
        img_k = compress_image(U, sigma, Vt, k)
        plt.subplot(2, 3, i+2)
        plt.imshow(img_k, cmap='gray')
        plt.title(f'k={k}')
        plt.axis('off')

        # 计算压缩率（存储空间对比）
        original_size = img.size  # 总像素数
        compressed_size = U[:, :k].size + sigma[:k].size + Vt[:k, :].size
        compression_ratio = compressed_size / original_size
        print(f"k={k}: 压缩后大小 = {compressed_size}, 压缩率 = {compression_ratio:.2%}")

    # 显示
    plt.tight_layout()
    plt.savefig("compressed_image.png")

    def find_k_for_energy(sigma,energy_threshold=0.9):
        total_energy = np.sum(sigma**2)
        cumsum_energy = np.cumsum(sigma**2)
        k = np.argmax(cumsum_energy >= total_energy * energy_threshold) +1
        return k


    # 加载彩色图像
    img_color = face()  # 彩色图 (H, W, 3)
    print(f"彩色图像形状: {img_color.shape}")

    # 分配存储
    compressed_color = np.zeros_like(img_color)

    # 对每个通道分别 SVD
    for i in range(3):  # R, G, B
        U, sigma, Vt = linalg.svd(img_color[:, :, i], full_matrices=False)
        k = 50  # 可调整
        compressed_color[:, :, i] = compress_image(U, sigma, Vt, k)

    # 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_color)
    plt.title("原图")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(compressed_color, 0, 255).astype(np.uint8))  # 限制范围
    plt.title(f"彩色图像压缩 (k=50)")
    plt.axis('off')

    plt.show()
