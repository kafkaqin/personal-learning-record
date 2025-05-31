from bs4 import BeautifulSoup
import json
import requests
import os
from Crypto.Cipher import AES
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://fsfsfs/"
}
session = requests.Session()
session.headers.update(HEADERS)
def get_video_url(page_url):

    resp = session.get(page_url)
    html = resp.text

    soup = BeautifulSoup(html, 'html.parser')

    # 找到 class 是 dplayer 的 div 标签
    div = soup.find('div', class_='dplayer')

    video_url = ""
    if div and div.has_attr('data-config'):
        data_config_str = div['data-config']
        # JSON 字符串里的反斜杠需要自动处理，json.loads 会自动解析转义字符
        data_config = json.loads(data_config_str)

        video_url = data_config.get('video', {}).get('url')
        print(video_url)
    else:
        print("没有找到 dplayer 的 data-config 属性")

    return video_url

import re

def parse_m3u8(m3u8_text: str):
    key_uri = None
    iv = None
    ts_urls = []

    lines = m3u8_text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("#EXT-X-KEY"):
            # 提取 URI 和 IV
            uri_match = re.search(r'URI="([^"]+)"', line)
            iv_match = re.search(r'IV=0x([0-9a-fA-F]+)', line)
            if uri_match:
                key_uri = uri_match.group(1)
            if iv_match:
                iv = bytes.fromhex(iv_match.group(1))
        elif line and not line.startswith("#"):
            ts_urls.append(line)

    return key_uri, iv, ts_urls

# 示例用法：读取 m3u8 内容
page_url = "https://fcccc"
m3u8_url = get_video_url(page_url)
print("m3u8_url====>",m3u8_url)
resp = requests.get(m3u8_url)
m3u8_text = resp.text

key_uri, iv, ts_urls = parse_m3u8(m3u8_text)

print("🔐 KEY URI:", key_uri)
print("🧪 IV:", iv.hex() if iv else None)
print("🎞️ TS URL 示例:", ts_urls[:3], f"...共 {len(ts_urls)} 个片段")

key = requests.get(key_uri).content
assert len(key) == 16, "密钥不是 128 位！"


# 创建输出目录
os.makedirs("ts", exist_ok=True)

# 解密并保存每个 TS 文件
for idx, url in enumerate(tqdm(ts_urls, desc="下载并解密")):
    ts_name = f"ts/seg_{idx:03d}.ts"

    # 下载 ts 加密文件
    resp = requests.get(url)
    encrypted_data = resp.content

    # 解密
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = cipher.decrypt(encrypted_data)

    # 保存解密后的 ts
    with open(ts_name, "wb") as f:
        f.write(decrypted_data)

# 合并所有 ts 为一个 mp4
output_file = "output1sd.ts"
with open(output_file, "wb") as out:
    for idx in range(len(ts_urls)):
        ts_path = f"ts/seg_{idx:03d}.ts"
        with open(ts_path, "rb") as f:
            out.write(f.read())

print(f"✅ 解密完成，已保存为 {output_file}")
os.system(f"ffmpeg -y -i {output_file} -c copy output_file111.mp4")