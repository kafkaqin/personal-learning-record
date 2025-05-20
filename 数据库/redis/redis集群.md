```shell
services:
  redis-node-1:
    image: redis:7.0
    container_name: redis-node-1
    ports:
      - 7001:7001
      - 17001:17001
    network_mode: "host"
    volumes:
      - redis-data-1:/data
    command: >
      sh -c "redis-server --port 7001 --cluster-enabled yes
             --cluster-config-file nodes.conf
             --cluster-node-timeout 5000
             --appendonly yes
             --bind 0.0.0.0
             --protected-mode no
             --cluster-announce-ip 192.168.31.129
             --cluster-announce-port 7001
             --cluster-announce-bus-port 17001"

  redis-node-2:
    image: redis:7.0
    container_name: redis-node-2
    ports:
      - 7002:7002
      - 17002:17002
    network_mode: "host"
    volumes:
      - redis-data-2:/data
    command: >
      sh -c "redis-server --port 7002 --cluster-enabled yes
             --cluster-config-file nodes.conf
             --cluster-node-timeout 5000
             --appendonly yes
             --bind 0.0.0.0
             --protected-mode no
             --cluster-announce-ip 192.168.31.129
             --cluster-announce-port 7002
             --cluster-announce-bus-port 17002"
  redis-node-3:
    image: redis:7.0
    container_name: redis-node-3
    ports:
      - 7003:7003
      - 17003:17003
    network_mode: "host"
    volumes:
      - redis-data-3:/data
    command: >
      sh -c "redis-server --port 7003 --cluster-enabled yes
             --cluster-config-file nodes.conf
             --cluster-node-timeout 5000
             --appendonly yes
             --bind 0.0.0.0
             --protected-mode no
             --cluster-announce-ip 192.168.31.129
             --cluster-announce-port 7003
             --cluster-announce-bus-port 17003"

  redis-node-4:
    image: redis:7.0
    container_name: redis-node-4
    ports:
      - 7004:7004
      - 17004:17004
    network_mode: "host"
    volumes:
      - redis-data-4:/data
    command: >
      sh -c "redis-server --port 7004 --cluster-enabled yes
             --cluster-config-file nodes.conf
             --cluster-node-timeout 5000
             --appendonly yes
             --bind 0.0.0.0
             --protected-mode no
             --cluster-announce-ip 192.168.31.129
             --cluster-announce-port 7004
             --cluster-announce-bus-port 17004"

  redis-node-5:
    image: redis:7.0
    container_name: redis-node-5
    ports:
      - 7005:7005
      - 17005:17005
    network_mode: "host"
    volumes:
      - redis-data-5:/data
    command: >
      sh -c "redis-server --port 7005 --cluster-enabled yes
             --cluster-config-file nodes.conf
             --cluster-node-timeout 5000
             --appendonly yes
             --bind 0.0.0.0
             --protected-mode no
             --cluster-announce-ip 192.168.31.129
             --cluster-announce-port 7005
             --cluster-announce-bus-port 17005"

  redis-node-6:
    image: redis:7.0
    container_name: redis-node-6
    ports:
      - 7006:7006
      - 17006:17006
    network_mode: "host"
    volumes:
      - redis-data-6:/data
    command: >
      sh -c "redis-server --port 7006 --cluster-enabled yes
             --cluster-config-file nodes.conf
             --cluster-node-timeout 5000
             --appendonly yes
             --bind 0.0.0.0
             --protected-mode no
             --cluster-announce-ip 192.168.31.129
             --cluster-announce-port 7006
             --cluster-announce-bus-port 17006"

  redis-setup:
    image: redis:7.0
    container_name: redis-setup
    network_mode: "host"
    depends_on:
      - redis-node-1
      - redis-node-2
      - redis-node-3
      - redis-node-4
      - redis-node-5
      - redis-node-6
    command: >
      sh -c 'for i in $(seq 1 30); do \
                redis-cli -h 192.168.31.129 -p 7001 ping > /dev/null 2>&1 && break; \
                echo "等待 Redis 节点启动..."; \
                sleep 1; \
              done; \
              redis-cli --cluster create \
                192.168.31.129:7001 \
                192.168.31.129:7002 \
                192.168.31.129:7003 \
                192.168.31.129:7004 \
                192.168.31.129:7005 \
                192.168.31.129:7006 \
                --cluster-replicas 1 \
                --cluster-yes;sleep 1000;'

volumes:
  redis-data-1:
  redis-data-2:
  redis-data-3:
  redis-data-4:
  redis-data-5:
  redis-data-6:
```