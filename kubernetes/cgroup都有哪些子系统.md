cgroup（Control Groups）是 Linux 内核的一个功能，它允许你为系统中的一组进程限定资源使用情况，并对其进行限制、隔离和监控。cgroup 子系统专门针对不同类型的资源进行管理和控制。以下是 cgroup 主要的子系统及其功能简介：

### 1. **cpu**
- **功能**：控制 CPU 时间分配。
- **关键文件**：
    - `cpu.shares`: 定义 CPU 使用的相对权重。
    - `cpu.cfs_period_us` 和 `cpu.cfs_quota_us`: 控制完全公平调度器（CFS）下的 CPU 使用率上限。

### 2. **cpuset**
- **功能**：将进程绑定到特定的 CPU 和内存节点上，主要用于 NUMA 系统。
- **关键文件**：
    - `cpuset.cpus`: 指定哪些 CPU 核心可以被该 cgroup 中的进程使用。
    - `cpuset.mems`: 指定哪些内存节点可以被该 cgroup 中的进程访问。

### 3. **memory**
- **功能**：限制和报告内存使用，包括物理内存和交换空间。
- **关键文件**：
    - `memory.limit_in_bytes`: 设置最大可用内存量。
    - `memory.usage_in_bytes`: 当前使用的内存量。
    - `memory.failcnt`: 达到内存限制而失败的次数。

### 4. **blkio**
- **功能**：控制块设备（如硬盘、SSD）的输入输出操作。
- **关键文件**：
    - `blkio.weight`: 设置 I/O 权重，影响 I/O 调度优先级。
    - `blkio.throttle.read_bps_device`: 限制读取速率。

### 5. **devices**
- **功能**：控制对设备的访问权限。
- **关键文件**：
    - `devices.allow`: 允许访问的设备列表。
    - `devices.deny`: 拒绝访问的设备列表。

### 6. **net_cls**
- **功能**：标记网络数据包以便于流量控制或过滤。
- **关键文件**：
    - `net_cls.classid`: 分配给数据包的类标识符，用于 tc (traffic control) 工具。

### 7. **net_prio**
- **功能**：设置网络接口的优先级。
- **关键文件**：
    - `net_prio.ifpriomap`: 映射网络接口与优先级级别。

### 8. **freezer**
- **功能**：暂停和恢复一组进程。
- **关键文件**：
    - `freezer.state`: 可以设置为 FROZEN 或 THAWED，分别表示冻结状态或解冻状态。

### 9. **hugetlb**
- **功能**：管理 Huge Pages 的使用。
- **关键文件**：
    - `hugetlb.<size>.limit_in_bytes`: 限制 huge page 的使用量。

### 10. **perf_event**
- **功能**：允许性能监测工具只在特定的 cgroup 中收集数据。

### 11. **pids**
- **功能**：限制 cgroup 中可以创建的最大进程数。
- **关键文件**：
    - `pids.max`: 最大允许的进程数量。
    - `pids.current`: 当前的进程数量。

这些子系统共同作用，使得管理员能够精确地控制和管理各个进程组的资源使用，从而提高系统的稳定性和安全性。特别是在容器化环境中，像 Docker 和 Kubernetes 这样的平台利用了 cgroup 来实现对容器资源的有效隔离和限制。