import hashlib
data = b'he'

hash_obj = hashlib.sha256()

hash_obj.update(data)
hash_obj.update(b'llo')

hex_dig = hash_obj.hexdigest()
print(hex_dig)

data1 = b"secret_password"
data2 = b"secret_password"

hash1 = hashlib.sha256(data1).hexdigest()
hash2 = hashlib.sha256(data2).hexdigest()
print(hash1==hash2)