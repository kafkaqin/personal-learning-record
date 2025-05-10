# xxtea.py — 纯 Python 实现，可用于加密解密与前端对接一致
import struct
import urllib.parse
from base64 import b64encode, b64decode


def to_bytes(s: str) -> bytes:
    return s.encode('utf-8')


def to_string(b: bytes) -> str:
    return b.decode('utf-8', errors='ignore')


def pad_key(key: str) -> bytes:
    b = to_bytes(key)
    return b.ljust(16, b'\0')[:16]


def to_uint32_array(data: bytes, include_length: bool) -> list[int]:
    length = len(data)
    n = (length + 3) // 4
    result = list(struct.unpack('<' + 'I' * n, data.ljust(n * 4, b'\0')))
    if include_length:
        result.append(length)
    return result


def from_uint32_array(data: list[int], include_length: bool) -> bytes:
    if include_length:
        length = data[-1]
        data = data[:-1]
    else:
        length = len(data) * 4
    return struct.pack('<' + 'I' * len(data), *data)[:length]


def xxtea_encrypt(v: list[int], k: list[int]) -> list[int]:
    if len(v) < 2:
        return v
    n = len(v)
    delta = 0x9E3779B9
    q = 6 + 52 // n
    sum_val = 0
    z = v[-1]
    for _ in range(q):
        sum_val = (sum_val + delta) & 0xffffffff
        e = (sum_val >> 2) & 3
        for p in range(n):
            y = v[(p + 1) % n]
            mx = (((z >> 5) ^ (y << 2)) + ((y >> 3) ^ (z << 4))) ^ ((sum_val ^ y) + (k[(p & 3) ^ e] ^ z))
            v[p] = (v[p] + mx) & 0xffffffff
            z = v[p]
    return v


def xxtea_decrypt(v: list[int], k: list[int]) -> list[int]:
    if len(v) < 2:
        return v
    n = len(v)
    delta = 0x9E3779B9
    q = 6 + 52 // n
    sum_val = (q * delta) & 0xffffffff
    y = v[0]
    for _ in range(q):
        e = (sum_val >> 2) & 3
        for p in reversed(range(n)):
            z = v[(p - 1 + n) % n]
            mx = (((z >> 5) ^ (y << 2)) + ((y >> 3) ^ (z << 4))) ^ ((sum_val ^ y) + (k[(p & 3) ^ e] ^ z))
            v[p] = (v[p] - mx) & 0xffffffff
            y = v[p]
        sum_val = (sum_val - delta) & 0xffffffff
    return v


def encrypt(text: str, key: str) -> str:
    # text = urllib.parse.quote(original)
    data_bytes = to_bytes(text)
    key_bytes = pad_key(key)
    v = to_uint32_array(data_bytes, include_length=True)
    k = to_uint32_array(key_bytes, include_length=False)
    encrypted = xxtea_encrypt(v.copy(), k)
    encrypted_bytes = from_uint32_array(encrypted, include_length=False)
    return urllib.parse.quote(b64encode(encrypted_bytes).decode('utf-8'))


def decrypt(url_encoded: str, key: str) -> str:
    base64_text = urllib.parse.unquote(url_encoded)
    encrypted_bytes = b64decode(base64_text)
    key_bytes = pad_key(key)
    v = to_uint32_array(encrypted_bytes, include_length=False)
    k = to_uint32_array(key_bytes, include_length=False)
    decrypted = xxtea_decrypt(v.copy(), k)
    decrypted_bytes = from_uint32_array(decrypted, include_length=True)
    return to_string(decrypted_bytes)


if __name__ == '__main__':
    key = "enduresurv1ve"
    result = encrypt("pageNum=1&pageSize=10", key)
    print(result)
    # plain = decrypt("B1osLScyCWlNI5SFrG6q1pORV7iAHxCPgeb%2BZw%3D%3D", key)
    plain = decrypt(result, key)
    print("明文:", plain)
