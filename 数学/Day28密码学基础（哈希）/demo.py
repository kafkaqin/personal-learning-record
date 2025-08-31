import hashlib

data = b'hello'
hash_object = hashlib.sha256(data)
print(hash_object.hexdigest())
data.decode('utf-8')

text="hello"
hash_hex = hashlib.sha256(text.encode('UTF-8')).hexdigest()
print(hash_hex)

def hash_file(filename):
    sha256=hashlib.sha256()
    with open(filename, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
password="mypaasword"
salt="randomsalt123"
hashed=hashlib.sha256((password+salt).encode("UTF-8")).hexdigest()
print(hashed)

def left_rotate(n,b):
    return ((n<<b) |(n>>(32-b))) & 0xFFFFFFFF

def sha256_manual(message:bytes)->str:
    original_length=len(message)*8
    message+=b'\x80'
    while (len(message)%8)%512!=448:
        message+=b'\x00'
    message+=original_length.to_bytes(8,'big')
    # Step 2: 初始化哈希值（前8个质数的平方根的小数部分取前32位）
    h = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    ]

    # Step 3: 扩展常量（前64个质数的立方根的小数部分）
    k = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    chunks = [message[i:i+64] for i in range(0, len(message), 64)]
    for chunk in chunks:
        w = [int.from_bytes(chunk[i:i+4],'big') for i in range(0, 64, 4)] +[0]*48

        for i in range(16,64):
            s0 = left_rotate(w[i - 15], 7) ^ left_rotate(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = left_rotate(w[i - 2], 17) ^ left_rotate(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & 0xFFFFFFFF

# 初始化8个工作变量
        a, b, c, d, e, f, g, h = h

        # 主循环
        for i in range(64):
            S1 = left_rotate(e, 6) ^ left_rotate(e, 11) ^ left_rotate(e, 25)
            ch = (e & f) ^ ((~e) & g)
            temp1 = (h + S1 + ch + k[i] + w[i]) & 0xFFFFFFFF
            S0 = left_rotate(a, 2) ^ left_rotate(a, 13) ^ left_rotate(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF

            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF

        # 更新哈希值
        h[0] = (h[0] + a) & 0xFFFFFFFF
        h[1] = (h[1] + b) & 0xFFFFFFFF
        h[2] = (h[2] + c) & 0xFFFFFFFF
        h[3] = (h[3] + d) & 0xFFFFFFFF
        h[4] = (h[4] + e) & 0xFFFFFFFF
        h[5] = (h[5] + f) & 0xFFFFFFFF
        h[6] = (h[6] + g) & 0xFFFFFFFF
        h[7] = (h[7] + h) & 0xFFFFFFFF

    # Step 5: 输出
    return ''.join(f'{val:08x}' for val in h)

# 测试手动实现
test_msg = b"hello"
manual_hash = sha256_manual(test_msg)
built_in_hash = hashlib.sha256(test_msg).hexdigest()

print(f"输入: {test_msg}")
print(f"手动实现: {manual_hash}")
print(f"hashlib:   {built_in_hash}")
print(f"匹配: {manual_hash == built_in_hash}")