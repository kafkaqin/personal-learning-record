import ctypes
import os

# 定义 ptrace 请求类型
PTRACE_PEEKDATA = 2
PTRACE_POKEDATA = 5
PTRACE_ATTACH = 16
PTRACE_DETACH = 17

# 加载 libc 库
libc = ctypes.CDLL("libc.so.6")

# 定义 ptrace 函数原型
def ptrace(request, pid, addr, data):
    return libc.ptrace(request, pid, addr, data)

# 目标进程ID (替换为实际PID)
pid = 6740
# 想要读取/写入的内存地址 (替换为实际地址)
address = 0x1000000

# 附加到目标进程
if ptrace(PTRACE_ATTACH, pid, None, None) == -1:
    raise OSError("Failed to attach to the process")
os.waitpid(pid, 0)  # 等待目标进程停止

# 读取数据
data = ctypes.c_long()
if ptrace(PTRACE_PEEKDATA, pid, address, ctypes.byref(data)) == -1:
    raise OSError("Failed to read memory")
print(f"原始值: {data.value}")

# 修改内存中的值
new_value = 2025
if ptrace(PTRACE_POKEDATA, pid, address, new_value) == -1:
    raise OSError("Failed to write memory")
print(f"已将值修改为: {new_value}")

# 再次读取以确认修改
if ptrace(PTRACE_PEEKDATA, pid, address, ctypes.byref(data)) == -1:
    raise OSError("Failed to re-read memory")
print(f"修改后的值: {data.value}")

# 分离目标进程
if ptrace(PTRACE_DETACH, pid, None, None) == -1:
    raise OSError("Failed to detach from the process")