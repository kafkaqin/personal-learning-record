调整 cgroup 子系统的资源限制主要是通过修改对应 cgroup 的控制文件来实现的。每个 cgroup 子系统都有特定的控制文件位于 `/sys/fs/cgroup/<subsystem>/` 目录下，这些文件允许你查看当前设置以及更改资源限制。以下是关于如何调整一些常见 cgroup 子系统的资源限制的基本步骤和示例。

### 基本步骤

1. **挂载 cgroup 文件系统**（如果尚未挂载）：
    - 大多数现代 Linux 发行版默认会挂载 cgroup 文件系统。如果没有，你可以手动挂载：
      ```bash
      sudo mount -t cgroup2 cgroup2 /sys/fs/cgroup/unified
      ```
   或者针对特定子系统：
     ```bash
     sudo mount -t cgroup -o cpu,cpuacct none /sys/fs/cgroup/cpu,cpuacct
     ```

2. **创建一个新的 cgroup**（如果你需要一个新组来应用不同的限制）：
    - 例如，要为 CPU 子系统创建一个名为 `mygroup` 的新组：
      ```bash
      sudo mkdir /sys/fs/cgroup/cpu/mygroup
      ```

3. **调整资源限制**：根据需要编辑相应的控制文件。

4. **将进程添加到 cgroup** 中：
    - 使用 `tasks` 文件可以将进程 ID 添加到指定的 cgroup 中：
      ```bash
      echo <PID> | sudo tee /sys/fs/cgroup/cpu/mygroup/tasks
      ```

### 示例

#### 1. CPU 子系统

- **设置 CPU 权重** (`cpu.shares`)：
    - `cpu.shares` 定义了 CPU 时间分配的相对权重，默认值是 1024。
      ```bash
      echo 512 | sudo tee /sys/fs/cgroup/cpu/mygroup/cpu.shares
      ```
  这表示该组内的进程将获得比默认少一半的 CPU 时间。

- **设置 CPU 使用率上限** (`cpu.cfs_period_us`, `cpu.cfs_quota_us`)：
    - 设置每 100ms 内最多使用 50ms 的 CPU 时间：
      ```bash
      echo 100000 | sudo tee /sys/fs/cgroup/cpu/mygroup/cpu.cfs_period_us
      echo 50000 | sudo tee /sys/fs/cgroup/cpu/mygroup/cpu.cfs_quota_us
      ```

#### 2. Memory 子系统

- **设置内存限制** (`memory.limit_in_bytes`)：
    - 设置最大可用内存量为 512MB：
      ```bash
      echo 512M | sudo tee /sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes
      ```

- **查看当前使用的内存量** (`memory.usage_in_bytes`)：
    - 查看当前使用的内存量：
      ```bash
      cat /sys/fs/cgroup/memory/mygroup/memory.usage_in_bytes
      ```

#### 3. Block I/O 子系统 (blkio)

- **设置读写速率限制** (`blkio.throttle.read_bps_device`, `blkio.throttle.write_bps_device`)：
    - 限制对设备 `8:0`（通常是第一个硬盘）的读取速率为 1MB/s：
      ```bash
      echo "8:0 1048576" | sudo tee /sys/fs/cgroup/blkio/mygroup/blkio.throttle.read_bps_device
      ```
    - 类似地，可以设置写入速率限制。

#### 4. Devices 子系统

- **允许或拒绝访问设备** (`devices.allow`, `devices.deny`)：
    - 允许访问所有块设备：
      ```bash
      echo 'a *:* rwm' | sudo tee /sys/fs/cgroup/devices/mygroup/devices.allow
      ```
    - 拒绝访问字符设备 `/dev/null`：
      ```bash
      echo 'c 1:3 w' | sudo tee /sys/fs/cgroup/devices/mygroup/devices.deny
      ```

### 注意事项

- 修改 cgroup 控制文件通常需要 root 权限。
- 在修改任何配置之前，请确保理解各个参数的意义及其可能带来的影响。
- 对于容器环境如 Docker 或 Kubernetes，它们通常提供了更高层次的抽象接口来管理 cgroup 资源限制，而不是直接操作 cgroup 文件系统。

通过上述方法，您可以有效地管理和调整 cgroup 下各子系统的资源限制，以满足不同应用场景的需求。