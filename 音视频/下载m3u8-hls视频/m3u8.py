import os
import requests
import m3u8
from tqdm import tqdm
from urllib.parse import urljoin

M3U8_URL = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
SESSION = requests.Session()
SAVE_DIR = "video_parts"
OUTPUT_FILE = "output1.ts"

os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: 加载 master playlist
print(f"正在下载主 m3u8 文件: {M3U8_URL}")
resp = SESSION.get(M3U8_URL)
master_playlist = m3u8.loads(resp.text)

if master_playlist.is_variant==False:
    print("未发现任何子播放列表（media playlist），请确认 URL 是否正确。")
    exit()

if not master_playlist.playlists:
    print("未发现任何子播放列表（media playlist），请确认 URL 是否正确。")
    exit()

print("master_playlist.playlists:",master_playlist.playlists)
# 选第一个清晰度
variant = master_playlist.playlists[0]
variant_uri = urljoin(M3U8_URL, variant.uri)
print(f"使用清晰度子播放列表: {variant_uri}")

# Step 2: 加载 media playlist，获取 .ts 片段
media_resp = SESSION.get(variant_uri)
media_playlist = m3u8.loads(media_resp.text)

segments = media_playlist.segments
print(f"共找到 {len(segments)} 个分片，开始下载...")

# 下载每个 ts 分片
for i, segment in enumerate(tqdm(segments)):
    ts_url = urljoin(variant_uri, segment.uri)
    ts_path = os.path.join(SAVE_DIR, f"{i:04}.ts")
    if not os.path.exists(ts_path):
        r = SESSION.get(ts_url)
        with open(ts_path, 'wb') as f:
            f.write(r.content)

# 合并所有 ts 分片为 output.ts
with open(OUTPUT_FILE, "wb") as merged:
    for i in range(len(segments)):
        ts_path = os.path.join(SAVE_DIR, f"{i:04}.ts")
        with open(ts_path, "rb") as part:
            merged.write(part.read())

print("合并完成，接下来转码为 MP4（可选）")

# 转码
os.system(f"ffmpeg -y -i {OUTPUT_FILE} -c copy output1.mp4")
