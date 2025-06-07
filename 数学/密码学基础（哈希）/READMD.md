
当然可以！我们可以使用 Python 内置的 `hashlib` 模块来实现 **SHA-256 哈希函数**。这是最简单且推荐的方式。

---

## ✅ 一、使用 `hashlib` 实现 SHA-256（推荐）

### 🧪 示例代码：

```python
import hashlib

# 输入数据（必须是字节类型）
data = b'hello'

# 创建 SHA-256 哈希对象
hash_obj = hashlib.sha256()

# 更新哈希对象（可以多次调用）
hash_obj.update(data)

# 获取十六进制格式的摘要
hex_digest = hash_obj.hexdigest()

print("SHA-256 Hash:", hex_digest)
```

### 🔍 输出结果：

```
SHA-256 Hash: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9826
```

---

## 🧠 小贴士：

- `b'hello'` 表示字节字符串（bytes），是 `hashlib` 要求的输入格式。
- 如果你想哈希字符串，需要先进行编码（通常是 UTF-8）：

```python
data = "你好，世界".encode('utf-8')
hash_obj = hashlib.sha256(data)
print(hash_obj.hexdigest())
```

---

## ✅ 二、多行更新（Update 多次）

你可以多次调用 `.update()` 方法，适合处理大文件或流式数据：

```python
hash_obj = hashlib.sha256()
hash_obj.update(b'he')
hash_obj.update(b'llo')
print(hash_obj.hexdigest())
```

输出与直接 `update(b'hello')` 相同。

---

## ✅ 三、验证两个数据是否相同（比较哈希值）

```python
data1 = b"secret_password"
data2 = b"secret_password"

hash1 = hashlib.sha256(data1).hexdigest()
hash2 = hashlib.sha256(data2).hexdigest()

if hash1 == hash2:
    print("数据一致")
else:
    print("数据不同")
```

---

## ✅ 四、SHA-256 的常见用途

| 应用场景 | 说明 |
|----------|------|
| 密码存储 | 存储密码的哈希而非明文（通常配合 salt 使用） |
| 数据完整性校验 | 下载文件后通过比对 SHA-256 校验码确认未被篡改 |
| 区块链 | Bitcoin 等区块链系统广泛使用 SHA-256 进行交易哈希和挖矿 |
| 数字签名 | 用于生成消息摘要，再进行加密签名 |

---

## 🚫 注意事项

- SHA-256 是一种**单向哈希函数**，不可逆。
- 不要直接用 SHA-256 存储用户密码，建议使用更安全的方法如 `bcrypt`, `scrypt` 或 `PBKDF2`。

---