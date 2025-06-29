![img.png](img.png)

### webRTC通话 原理

#### 会议ID(房间号 频道)
1. 如何发现 信令服务器 信令通道
![img_1.png](img_1.png)
![img_2.png](img_2.png)
![img_3.png](img_3.png)
2. 如何发现（加入 离开 通知）
3. 音视频编解码能力沟通
浏览器: 
苹果浏览器
微软浏览器
谷歌浏览器
![img_4.png](img_4.png)
通过信令服务器协商 音视频编解码能力沟通

发起人(offer)： 
Answer
![img_6.png](img_6.png)
![img_5.png](img_5.png)
![img_7.png](img_7.png)
![img_8.png](img_8.png)

#### SDP(Session Description Protocol)媒体协商
![img_9.png](img_9.png)
![img_10.png](img_10.png)


#### 网络传输
NAT(Net Address Transport)
![img_11.png](img_11.png)
![img_12.png](img_12.png)

1. 为什么需要NAT 
  公网IP是有限的，如果没有NAT机制 就需要100个公网IP
  网络攻击 如果是独立的公网IP 公司的网络比较容易被攻击
2. 不同的网络，音视频数据如何转发
  两个外网IP如何互通
![img_13.png](img_13.png)
![img_14.png](img_14.png)
![img_15.png](img_15.png)
![img_16.png](img_16.png)
![img_17.png](img_17.png)

#### 总结 信令服务器
1. 连接管理，客户端都需要连接到信令服务器
2. 房间管理,join(需要彼此通话，加入同样的房间) notify(其他人进房间 离开房间)  leave
3. 相互转发 SDP(封装了编码信息,实际上还有其他的信息),有offer answer
4. 相互转发candidate(封装了自己的网络信息)
5. 心跳包(用于信令机制的完善)
6. .....其他的一些完善工作
#### Candidate(封装了网络信息
![img_18.png](img_18.png))

#### 检测浏览器是否支持webRTC
https://web.sdk.qcloud.com/trtc/webrtc/demo/detect/index.html

#### turn/stun的服务是独立的
![img_19.png](img_19.png)


## 代码解读
![img_20.png](img_20.png)

客户端代码
![img_21.png](img_21.png)
![img_22.png](img_22.png)
![img_23.png](img_23.png)
![img_24.png](img_24.png)
![img_25.png](img_25.png)
![img_26.png](img_26.png)
![img_27.png](img_27.png)
![img_28.png](img_28.png)
![img_29.png](img_29.png)
![img_30.png](img_30.png)

RTCPeerConnection
![img_31.png](img_31.png)
![img_32.png](img_32.png)
![img_33.png](img_33.png)
![img_34.png](img_34.png)
![img_35.png](img_35.png)
![img_36.png](img_36.png)
![img_37.png](img_37.png)
![img_38.png](img_38.png)
