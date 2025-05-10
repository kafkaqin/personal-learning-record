import numpy as np

# 时间域转换到频率域
if __name__ == '__main__':
    A = np.array([[4,7],[2,6]])
    print("原始矩阵:\n",A)
    det_A = np.linalg.det(A)
    print("\n矩阵的行列式:{:.2f}".format(det_A))

    if det_A != 0:
        inv_A = np.linalg.inv(A)
        print("\n矩阵 A 的逆矩阵:\n",inv_A)

        identity_check = np.round(A @ inv_A,decimals=10)
        print("\n 验证 A * inv(A) ~~单位矩阵:\n",identity_check)
    else:
        print("矩阵不可逆")
    I = np.eye(2)
    is_close = np.allclose(A @ inv_A,I)
    print("A * inv(A) 是否接近单位矩阵?:",is_close)