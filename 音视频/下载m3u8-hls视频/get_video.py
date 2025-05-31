import os
import re
import requests
from urllib.parse import urljoin
from pathlib import Path
import subprocess

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://xx.com/"
}

BASE_URL = "https://xxx.com"

session = requests.Session()
session.headers.update(HEADERS)

def download_m3u8(m3u8_url, base_path, output_filename):
    """ 下载 M3U8 音视频内容（含 init 段）合并为 TS/MP4 """
    resp = session.get(m3u8_url)
    lines = resp.text.strip().splitlines()
    init_url = None
    segments = []

    for line in lines:
        if line.startswith("#EXT-X-MAP"):
            match = re.search(r'URI="([^"]+)"', line)
            if match:
                init_url = urljoin(m3u8_url, match.group(1))
        elif not line.startswith("#") and line.endswith(".m4s"):
            segments.append(urljoin(m3u8_url, line))

    file_path = os.path.join(base_path, output_filename)
    with open(file_path, "wb") as f:
        if init_url:
            print(f"Downloading init segment: {init_url}")
            f.write(session.get(init_url).content)
        for seg_url in segments:
            print(f"Downloading segment: {seg_url}")
            f.write(session.get(seg_url).content)

    return file_path

def extract_urls(master_m3u8_text):
    audio_urls = {}
    video_urls = {}

    for line in master_m3u8_text.strip().splitlines():
        if line.startswith("#EXT-X-MEDIA") and "AUDIO" in line:
            group_id = re.search(r'GROUP-ID="([^"]+)"', line).group(1)
            uri = re.search(r'URI="([^"]+)"', line).group(1)
            audio_urls[group_id] = urljoin(BASE_URL, uri)
        elif line.startswith("#EXT-X-STREAM-INF"):
            match = re.search(r'AUDIO="([^"]+)"', line)
            group_id = match.group(1) if match else None
        elif line.startswith("/") and group_id:
            video_urls[group_id] = urljoin(BASE_URL, line)

    return audio_urls, video_urls

def merge_audio_video(video_file, audio_file, output_file):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_file,
        "-i", audio_file,
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_file
    ]
    subprocess.run(cmd, check=True)

def main():
    # 你从 requests.text 拿到的 master m3u8 内容
    M3U8_URL = ""
    print(f"正在下载主 m3u8 文件: {M3U8_URL}")
    resp = requests.session().get(M3U8_URL)

    audio_urls, video_urls = extract_urls(resp.text)

    # 你可以更换这里选择的清晰度
    group_id = "audio-128000"
    audio_m3u8 = audio_urls[group_id]
    video_m3u8 = video_urls[group_id]

    print(f"[INFO] 下载视频: {video_m3u8}")
    print(f"[INFO] 下载音频: {audio_m3u8}")

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    id=getUniqueID(M3U8_URL)
    video_file = download_m3u8(video_m3u8, temp_dir, f"{id}-video.mp4")
    audio_file = download_m3u8(audio_m3u8, temp_dir, f"{id}-audio.mp4")
    output_file = "final_output.mp4"

    merge_audio_video(video_file, audio_file, output_file)
    print(f"[SUCCESS] 合并完成: {output_file}")

def getUniqueID(url):
    pattern = r"/amplify_video/(\d+)/"
    match = re.search(pattern, url)
    value = ""
    if match:
        value = match.group(1)
    else:
        print("没找到匹配的数字")
    return value
if __name__ == "__main__":
    main()
    # URL = ""
    # resp = requests.session().get(URL)
    # pattern = r'<video[^>]*\sdata-source="([^"]+)"'
    # matches = re.findall(pattern, resp.text)
    #
    # for src in matches:
    #     print(src)
    # url = "xx"
    # print(getUniqueID(url))
