#### 1.停止和禁用开机启动systemd-resolved
```shell
systemctl stop systemd-resolved
systemctl disable systemd-resolved
```

#### 2.修改/etc/resolv.conf
```shell
#nameserver 127.0.0.53
nameserver 127.0.0.1
options edns0 trust-ad
search .
```

#### 3. 下载dnsmasq
```shell
apt-get update 
apt-get install dnsmasq
```
#### 4.修改dnsmasq配置文件(server指向 /run/systemd/resolve/resolv.conf文件)
```shell
vim /etc/dnsmasq.conf
```
```shell

root@iZwz9gr05c194c7pt5scygZ:~# cat /etc/dnsmasq.conf
# Configuration file for dnsmasq.
#
# Format is one option per line, legal options are the same
# as the long options legal on the command line. See
# "/usr/sbin/dnsmasq --help" or "man 8 dnsmasq" for details.

# Listen on this specific port instead of the standard DNS port
# (53). Setting this to zero completely disables DNS function,
# leaving only DHCP and/or TFTP.
port=53
listen-address=0.0.0.0
server=100.100.2.136
server=100.100.2.138
```
#### 5.重启和设置开启启动dnsmasq
```shell
systemctl restart dnsmasq
```

#### 6.验证
```shell
$nslookup lucksafeysmarket02.aa144.cn
Server:         127.0.0.1
Address:        127.0.0.1#53

Non-authoritative answer:
lucksafeysmarket02.aa144.cn     canonical name = lucksafeysmarket02.aa144.cn.eac6eea1.cdnhwcaoc115.cn.
lucksafeysmarket02.aa144.cn.eac6eea1.cdnhwcaoc115.cn    canonical name = ddos-eac6eea1d60244168a9ffc4210dd4101-web.svipgulr.com.
Name:   ddos-eac6eea1d60244168a9ffc4210dd4101-web.svipgulr.com
Address: 183.60.255.171
Name:   ddos-eac6eea1d60244168a9ffc4210dd4101-web.svipgulr.com
Address: 183.60.255.165
Name:   ddos-eac6eea1d60244168a9ffc4210dd4101-web.svipgulr.com
Address: 183.60.255.183
Name:   ddos-eac6eea1d60244168a9ffc4210dd4101-web.svipgulr.com
Address: 183.60.255.174

### 或者
dig @47.107.143.93 -p 53 lucksafeysmarket02.aa144.cn
```
#### kubernetes
https://help.aliyun.com/zh/ack/ack-managed-and-ack-dedicated/user-guide/configure-coredns


https://dyrnq.com/ubuntu-update-etc-resolve-conf/