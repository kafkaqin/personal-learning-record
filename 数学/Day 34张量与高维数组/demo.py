import numpy as np

vec = np.array([1,2,3])
print("1D向量 数组:",vec)

mat = np.array([[1,2],[3,4]])
print("\n 2D数组:",mat)

tensor_3d = np.array([
    [[1,2],[3,4]],
[[5,6],[7,8]]
])

print("\n 3D数组(张量):",tensor_3d)

arr = np.arange(24).reshape((2,3,4)) # 2块 3行4列

print("\n reshape 创建的 3D数组:\n",arr)

print("====ndarray的属性====")
print("形状(shape):",arr.shape)
print("维度(ndim):",arr.ndim)
print("元数总数(size):",arr.size)
print("数据类型(dtype):",arr.dtype)

print("====ndarray的索引====")
print("arr=[0,1,2]=",arr[0,1,2])
print("\narr=[0,:,:]=",arr[0,:,:])
print("\narr=[0,:2,:2]=",arr[0,:2,:2])

print("ndarray 广播")
arr_add = arr+10
print("加10后的数组",arr_add)

print("ndarray的沿轴操作")
sum_axis0 = np.sum(arr,axis=0)
print("\n 沿 axis=0 求和：\n",sum_axis0)

sum_axis1 = np.sum(arr,axis=1)
print("\n 沿 axis=1 求和：\n",sum_axis1)

print("ndarray的转置")
transposed = np.transpose((2,1,0))
print("\n转置后的shape",transposed.shape)


import numpy as np
arr_3d = np.array(
    [
        [[1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]],
        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]

    ]
)

print(f"形状:{arr_3d.shape}")
print(f"维度:{arr_3d.ndim}")
print(f"总元素:{arr_3d.size}")

zeros = np.zeros((2,3, 4,5))
ones = np.ones((3,3,3))
identity = np.eye(4)
arr_3d = np.arange(24).reshape(2,3,4)
random_4d = np.random.rand(2,2,3,3)
print("原始 3D数组:")
print(arr_3d)
block = arr_3d[1]
print(f"第一块 shape：{block.shape}")
print(f"第一块：{block}")
element = arr_3d[1,2,3]
print(f"取第一块第二行第三列:{element}")

slice_1 = arr_3d[:,:2,:]
print(f"取两个快的前两行，所有列:{slice_1.shape}")
slice_2 = arr_3d[::2,::2,::2]
print(f"维度取偶数索引{slice_2.shape}")

batch_size = 4
height = 32
width = 32
channels = 3
images = np.random.randint(0,256,size=(batch_size,height,width,channels),dtype=np.uint8)
print(f"图像批次形状:{images.shape}")
img0 = images[0]
reds = images[...,0]

ts_data = np.random.randn(100,50,10)
print(f"时序数据形状:{ts_data.shape}")
sample5_features = ts_data[4,:,:3]

W = np.random.randn(128,64)
kernel = np.random.randn(32,3,5,5)
gamma = np.random.randn(64)
beta = np.random.randn(64)

flat = np.arange(24)
reshape = flat.reshape(2,3,4)
back = reshape.reshape(-1)
arr_T = arr_3d.transpose(2,0,1)
print(arr_T)
bias = np.array([1,2,3])
expanded_bias = bias[np.newaxis,np.newaxis,:]
result = arr_3d+expanded_bias
mean_per_block = arr_3d.mean(axis=(1,2))
global_max = arr_3d.max()
sum_over_time = ts_data.sum(axis=1)
print(f"C连续？{arr_3d.flags['C_CONTIGUOUS']}")
print(f"F连续:?{arr_3d.flags['F_CONTIGUOUS']}")