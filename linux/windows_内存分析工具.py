import ctypes
from ctypes import wintypes

# 定义一些Windows API函数
OpenProcess = ctypes.windll.kernel32.OpenProcess
ReadProcessMemory = ctypes.windll.kernel32.ReadProcessMemory
WriteProcessMemory = ctypes.windll.kernel32.WriteProcessMemory
CloseHandle = ctypes.windll.kernel32.CloseHandle

# 定义常量
PROCESS_ALL_ACCESS = (0x000F0000 | 0x00100000 | 0xFFF)

# 目标进程ID（需替换为实际PID）
pid = 1234
# 想要读取/写入的内存地址（需替换为实际地址）
address = 0x1000000

# 打开目标进程
process_handle = OpenProcess(PROCESS_ALL_ACCESS, False, pid)
if not process_handle:
raise ctypes.WinError()

# 准备读取数据
buffer = ctypes.c_uint()
buffer_size = ctypes.sizeof(buffer)
bytes_read = ctypes.c_ulong(0)

# 读取内存
if ReadProcessMemory(process_handle, address, ctypes.byref(buffer), buffer_size, ctypes.byref(bytes_read)):
print(f"原始值: {buffer.value}")
else:
raise ctypes.WinError()

# 修改内存中的值
new_value = 2025
if WriteProcessMemory(process_handle, address, ctypes.byref(ctypes.c_uint(new_value)), buffer_size, None):
print(f"已将值修改为: {new_value}")
else:
raise ctypes.WinError()

# 再次读取以确认修改
if ReadProcessMemory(process_handle, address, ctypes.byref(buffer), buffer_size, ctypes.byref(bytes_read)):
print(f"修改后的值: {buffer.value}")
else:
raise ctypes.WinError()

# 关闭句柄
CloseHandle(process_handle)
