## 内容
1. zlmediakit编译
2. zlmediakit网络模型分析
3. zlmediakit RTSP 推拉流
4. 如何掌握zlmediakit的二次开发

## 音视频
1. 移动客户端的音视频
2. QT桌面音视频
3. 嵌入式音视频

## media-server
## ZLToolkit
![img.png](img.png)
![img_1.png](img_1.png)
![img_2.png](img_2.png)
![img_3.png](img_3.png)
![img_4.png](img_4.png)

![img_5.png](img_5.png)
![img_6.png](img_6.png)

```shell
lsof -i :80
```

![img_7.png](img_7.png)
![img_8.png](img_8.png)

### 抓包的技巧
1. 推流的包
![img_9.png](img_9.png)
![img_10.png](img_10.png)

2. 拉流的包
在拉流的客户端抓包
![img_11.png](img_11.png)
![img_12.png](img_12.png)
![img_13.png](img_13.png)
![img_14.png](img_14.png)
![img_15.png](img_15.png)
![img_16.png](img_16.png)

3. 网络模型
多线程 每一个线程绑定一个epoll
![img_17.png](img_17.png)
![img_18.png](img_18.png)
![img_19.png](img_19.png)
![img_20.png](img_20.png)
![img_21.png](img_21.png)

#### RTSP 推拉流
![img_22.png](img_22.png)
handleReq_Option
![img_23.png](img_23.png)
![img_24.png](img_24.png)
换行符
![img_25.png](img_25.png)

![img_26.png](img_26.png)
![img_27.png](img_27.png)
#### 拉流转发数据如何做
![img_28.png](img_28.png)
![img_29.png](img_29.png)
live/test
每一个epoll 都需要创建绑定一个RingReaderDispatcher
![img_30.png](img_30.png)
![img_31.png](img_31.png)
![img_32.png](img_32.png)


![img_33.png](img_33.png)