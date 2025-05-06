import numpy as np
from numpy.linalg import norm
## 使用numpy实现傅立叶变换
if __name__ == '__main__':
    # a = np.array([3,4])
    # b = np.array([1,2])
    # print("向量a:",a)
    # print("向量b:",b)
    # add_result = a+b
    # print("向量a+b=",add_result)
    # print("向量a-b=",a-b)
    # print("向量点积 a*b",np.dot(a,b))
    # a3 = np.array([3,4,0])
    # b3 = np.array([1,2,0])
    # result_cross = np.cross(a3,b3)
    # print("向量叉积(外积) a x b ",result_cross) ## 向量叉积
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print("向量a=",a)
    print("向量b=",b)
    print("向量加法:a+b=",a+b)
    print("向量减法:a-b=",a-b)
    print("向量点积(内积):a-b=",np.dot(a,b))
    print("向量叉积:axb=",np.cross(a,b))
    mod_a = norm(a)
    mod_b = norm(b)
    print("向量的模长:mod_a=",mod_a)
    print("向量的模长:mod_b=",mod_b)

    unit_a = a / mod_a
    unit_b = b / mod_b
    print("向量a的单位向量=",unit_a)
    print("向量b的单位向量=",unit_b)

    ## 计算两个向量之间的夹角
    cos_theta = np.dot(unit_a,unit_b)
    angle_rad = np.arccos(np.clip(cos_theta,-1.0,1.0))
    angle_deg = np.degrees(angle_rad)
    print("向量a和b的夹角(弧度)=",angle_deg)
'''
向量的叉积（也称为外积或向量积）是仅在三维空间中定义的一种运算。给定两个三维向量
𝑎 ⃗ a
  和
𝑏
⃗
b
 ，它们的叉积
𝑎
⃗
×
𝑏
⃗
a
 ×
b
  是一个与这两个向量都垂直的新向量。这个新向量的方向由右手定则确定，其大小等于以这两个向量为边的平行四边形的面积。
'''

