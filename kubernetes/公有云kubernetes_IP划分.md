### 1. Kubernetes Service IP 范围
| 服务类型 | 	IP 范围（CIDR）                                            |	说明 |
|----------|---------------------------------------------------------| ----- |
| Kubernetes Service | 	192.168.32.0/24 ~ 192.168.55.0/24 |	24 个 Kubernetes 服务段 | 
### 2. 区域 VPC 子网配置
#### 2.1 北美区域（us-west-1, us-east-1）
| Region    | 	VPC 名称	               | VPC CIDR |	AZ	| 子网名称 |	子网 CIDR |
|-----------|------------------------| ----- | ------ | ------- | ----- |
| us-west-1 | 	access-prd-us-west-1  |	192.168.0.0/26	| us-west-1a	| subnet-bastion	|192.168.0.0/28 |
| us-west-1 | 	access-uat-us-west-1	 | 192.168.6.0/26        | 	us-west-1b	          | subnet-bastion	       | 192.168.6.0/28        |
| us-east-1 | 	access-prd-us-east-1	 | 192.168.0.64/26       | 	us-east-1a	          | subnet-bastion	       | 192.168.0.64/28       |
| us-east-1 | 	access-uat-us-east-1	 | 192.168.6.64/26	      | us-east-1b	           | subnet-bastion        | 	192.168.6.64/28      | 
#### 2.2 中国区域（上海、广州等）
| Region	| VPC 名称	| VPC CIDR | 	AZ	| 子网名称	| 子网 CIDR |
| -----------	| -----------	| ----------- | 	-----------	| -----------	| ----------- |
|上海	|access-prd-cn-shanghai1	| 192.168.3.0/26	| 1	subnet-bastion	| 192.168.3.0/28 |
| 上海	|access-uat-cn-shanghai1	|192.168.16.0/26	| 2	subnet-bastion	| 192.168.16.0/28 |
| 广州	| access-prd-cn-guangzhou	| 10.255.3.0/26	5	| sunet-bastion	| 10.255.3.0/28 |
### 3. 子网 IP 范围细分
#### 3.1 192.168 段子网
| 子网 CIDR | 	可用 IP |  数量 | 	区域 |
| -----------| 	----------- |  ----------- | 	----------- |
| 192.168.0.0/26 ~ 192.168.5.192/26	| 24 |	Region |
| 192.168.6.0/26 ~ 192.168.11.192/26	| 24 |	Region |
| 192.168.12.0/26 ~ 192.168.12.192/26	| 4	| Region |
| 192.168.13.0/24 ~ 192.168.24.0/24	| 12	| Region |
#### 3.2 10.127 段子网（上海区域）
| 子网用途 |	CIDR	| 可用 IP 范围           | 	说明         |
| -----------| 	----------- |--------------------|-------------|
| node |	10.127.0.0/20| 	10.127.0.1 ~ 10.127.15.254 | 	4094 个 IP  | 
| pod	| 10.127.16.0/20| 	10.127.16.1 ~ 10.127.31.254 | 	4094 个 Pod |
| data endpoint |	10.127.32.0/25 |	10.127.32.1 ~ 10.127.32.127	| 126  个端点    |
| dmz |	10.127.32.128/25| 	10.127.32.128 ~ 10.127.32.254| 	126 个 DMZ  |
### 4. AWS 区域配置示例
| Region	| VPC 名称            |	AZ	| 子网类型	| CIDR	| 用途              |
| -----------|-------------------|  --------- | 	---------- | ----- |-----------------|
| us-east-1	| prd-pub-us-east-1 |	us-east-1a	| node	| 10.141.0.0/20 | 	节点子网           |
| us-east-1	|                   | 	us-east-1a | pod	| 10.141.16.0/20	| Pod   子网        |
| us-east-1	| 	                 | us-east-1a 	| data	| 10.141.32.0/25	 | 数据端点 |
### 5. 特殊场景配置
#### 5.1 Demo 环境 VPC 划分
| 环境	| VPC CIDR 范围 | 	说明    |
| ---- | --- |--------|
| Demo |	10.1.0.0/16 ~ 10.126.0.0/16 |	基础服务 VPC |
|| 10.127.0.0/16 ~ 10.140.0.0/16 |	扩展服务 VPC |
||10.141.0.0/16 ~ 10.167.0.0/ |	隔离测试 VPC |
#### 5.2 传统应用（Legacy-APP）
| 应用类型	       | VPC CIDR	       | 子网 CIDR	| 子网数量 |
|-------------|-----------------|--------| --------|
| Legacy-APP	 | 192.168.0.0/16	 | 192.168.0.0/24	 | 3         |
### 6. 注意事项
* 1. 部分表格中 “-” 表示未指定或默认值。
* 2. CIDR 块解析需注意网络地址和广播地址不可用（如 /25 实际可用 IP 为 126 个）。
* 3. 区域（Region）与可用区（AZ）需配合 VPC 实现多可用区高可用部署。