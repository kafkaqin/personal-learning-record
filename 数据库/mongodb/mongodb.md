```
version: '3.2'
 
services:
  # 服务名称
  mongodb1:
    # 使用最新的 mongodb 镜像
    image: mongo:latest
    # docker 服务启动时，自动启动 mongo 容器
    restart: always
    # 容器的名称
    container_name: mongo1
    # 宿主机中的目录和文件，映射容器内部的目录和文件
    volumes:
      - ./mongo-keyfile:/data/mongodb.key
    ports:
      # 宿主机的端口映射容器内的端口
      - 27014:27017
    environment:
      # 初始化一个 root 角色的用户 jobs 密码是 123456
      - MONGO_INITDB_ROOT_USERNAME=admin_user
      - MONGO_INITDB_ROOT_PASSWORD=admin_password
    # 使用创建的桥接网络，把各个 mongodb 容器连接在一起
    networks:
      - mongoNetwork
    # 启动容器时，在容器内部额外执行的命令
    # 其中 --replSet 参数后面的 mongos 是集群名称，这个很重要
    command: mongod --replSet mongos --keyFile /data/mongodb.key
    entrypoint:
      - bash
      - -c
      - |
        chmod 400 /data/mongodb.key
        chown 999:999 /data/mongodb.key
        exec docker-entrypoint.sh $$@
  
  mongodb2:
    image: mongo:latest
    restart: always
    container_name: mongo2
    volumes:
      - ./mongo-keyfile:/data/mongodb.key
    ports:
      - 27013:27017
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin_user
      - MONGO_INITDB_ROOT_PASSWORD=admin_password
    networks:
      - mongoNetwork
    command: mongod --replSet mongos --keyFile /data/mongodb.key
    entrypoint:
      - bash
      - -c
      - |
        chmod 400 /data/mongodb.key
        chown 999:999 /data/mongodb.key
        exec docker-entrypoint.sh $$@
 
  mongodb3:
    image: mongo:latest
    restart: always
    container_name: mongo3
    volumes:
      - ./mongo-keyfile:/data/mongodb.key
    ports:
      - 27012:27017
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin_user
      - MONGO_INITDB_ROOT_PASSWORD=admin_password
    networks:
      - mongoNetwork
    command: mongod --replSet mongos --keyFile /data/mongodb.key
    entrypoint:
      - bash
      - -c
      - |
        chmod 400 /data/mongodb.key
        chown 999:999 /data/mongodb.key
        exec docker-entrypoint.sh $$@
 
# 创建一个桥接网络，把各个 mongodb 实例连接在一起，该网络适用于单机
# 如果在不同的宿主机上，使用 docker swarm 需要创建 overlay 网络
networks:
  mongoNetwork:
    driver: bridge
```