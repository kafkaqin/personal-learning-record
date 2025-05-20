```shell
docker run -d --name emqx -e EMQX_DASHBOARD__DEFAULT_PASSWORD=admin -p 18083:18083 -p 1883:1883 -p 8083:8083 emqx:latest
```