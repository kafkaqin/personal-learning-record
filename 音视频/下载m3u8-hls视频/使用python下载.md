#### 下载python依赖包
```shell
pip install requests m3u8 tqdm

```
#### .m3u8 有两种类型：主播放列表（master）和 媒体播放列表（media）

1. Master Playlist 一般包含这些标签：
```m3u8
#EXTM3U
#EXT-X-STREAM-INF

```
2. Media Playlist 一般包含这些标签：
```m3u8
#EXTM3U
#EXT-X-TARGETDURATION
#EXTINF
#EXT-X-ENDLIST

```


#### Python 常用库：
requests / httpx（基础请求）

aiohttp（异步下载）

m3u8 + ffmpeg-python

youtube-dl / yt-dlp（支持数千网站）

selenium / playwright

#### 加密的 .m3u8 视频下载和解密


### 下载 
```shell
pip install playwright
playwright install
pip install -U yt-dlp
pip install requests m3u8 pycryptodome

```

### 合并音视频
```shell
ffmpeg -i video.mp4 -i audio.aac -c copy -map 0:v -map 1:a final_output.mp4

```