def simple_hash(data: bytes, output_size: int = 8) -> bytes:
    """
    一个简单的教学用哈希函数
    :param data: 输入数据（bytes）
    :param output_size: 输出哈希长度（字节）
    :return: 哈希值（bytes）
    """
    if len(data) == 0:
        return b'\x00' * output_size

    # 初始化状态（类似 SHA 的初始向量）
    state = [0x6A, 0x9E, 0x66, 0x7F, 0xBB, 0x67, 0xAE, 0x85]  # 取 SHA-256 初始值前8字节

    # 处理每个字节
    for i, byte in enumerate(data):
        # 用索引和字节值扰动状态
        for j in range(len(state)):
            # 异或 + 循环左移 + 加法
            shift_amount = (i + j + 1) % 7 + 1  # 1~7位
            rotated = ((state[j] << shift_amount) | (state[j] >> (8 - shift_amount))) & 0xFF
            state[j] = (rotated ^ byte ^ (i * 7 + j * 11)) & 0xFF

    # 混淆输出
    final = []
    for j in range(output_size):
        # 组合多个状态字节
        val = 0
        for k in range(len(state)):
            val ^= state[k] << ((j + k) % 4 * 2)  # 位移混合
        final.append(val % 256)

    return bytes(final)

# 测试
def test_hash():
    msg1 = b"hello"
    msg2 = b"hellp"  # 只改一个字母
    msg3 = b"Hello"

    h1 = simple_hash(msg1)
    h2 = simple_hash(msg2)
    h3 = simple_hash(msg3)

    print(f"输入: {msg1} -> 哈希: {h1.hex()}")
    print(f"输入: {msg2} -> 哈希: {h2.hex()}")
    print(f"输入: {msg3} -> 哈希: {h3.hex()}")

    # 检查雪崩效应
    diff1 = sum(bin(a ^ b).count('1') for a, b in zip(h1, h2))
    diff2 = sum(bin(a ^ b).count('1') for a, b in zip(h1, h3))
    print(f"hello vs hellp: {diff1} 位不同（雪崩效应）")
    print(f"hello vs Hello: {diff2} 位不同")

test_hash()


def fletcher_like_hash(data: bytes, size=8) -> bytes:
    """
    基于 Fletcher 校验和的简单哈希（增强扩散性）
    """
    if not data:
        return b'\x00' * size

    # 使用两个累加器
    sum1 = 1
    sum2 = 0
    prime = 257  # 小质数

    for byte in data:
        sum1 = (sum1 + byte) % prime
        sum2 = (sum2 + sum1) % prime

    # 混合输出
    combined = (sum1 << 16) | sum2
    output = []
    for _ in range(size):
        output.append(combined % 256)
        combined = (combined * 6364136223846793005 + 1) % (2**64)  # 简单随机化

    return bytes(output)

# 测试
print("Fletcher-like 哈希:")
print(f"hello:  {fletcher_like_hash(b'hello').hex()}")
print(f"hellp:  {fletcher_like_hash(b'hellp').hex()}")
print(f"hello!: {fletcher_like_hash(b'hello!').hex()}")