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