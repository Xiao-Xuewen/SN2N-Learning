import numpy as np
from numpy import pi
from numpy import newaxis
# print(np.__version__)

# a = np.array([10,20,30]) #将一个python列表转换成numpy一维数组

# print(a)

# img = np.array([[10,20,30,40],
#                 [20,30,40,50],
#                 [30,40,50,60]]) #二维数组，使用逗号分隔
# print(img) 
# print(img.shape) #shape = [H*W] 也就是行*列（高度*宽度）
# print(img.ndim) #查看数组维数
# print(img.size) #查看元素总数
# print(img.dtype) #查看数组中的元素数据类型

# #序列，序列组成二维数组 序列的序列，序列的序列构成三维数组
# print(np.zeros((3,4,5))) #创建一个三行四列矩阵数组 每个元素是包含五个0的数组
# print(np.ones((3,4))) #1
# print(np.empty((2,3)))#0

# print(np.arange(10,30,5))# start end step
# print(np.linspace(0,2,9))# 0-2 等步长 9个数
# #绘制sinx的数据
# x = np.linspace(0,2*pi,100)
# y = np.sin(x)

# #基本操作
# a = np.array([10,20,30,40])
# b = np.arange(4) # 默认start=0 step=1
# c = a - b 
# print(c)
# print(b**2)
# print(a < 25)

# a = np.array([[1,0],
#               [0,3]])
# b = np.array([[2,3],
#               [2,0]])
# print(a*b) #按位置相乘
# print(a@b) #矩阵乘法 或者 a.dot(b)

# rg = np.random.default_rng(1) #随机数生成器 1 是seed
# a = np.ones((2,3),dtype=np.int64)
# b = rg.random((2,3)) #默认float64
# print(a,b)
# b += a
# print(b)
# #a += b
# #print(a) #报错 原地操作会保护原有的数据类型
# c = a + b
# print(c,c.dtype)
# #操作不同类型的数组时，一般得到的数组类型是更精确的
# a = rg.random((3,4))
# print(a)
# print(a.sum())
# print(a.min())
# print(a.max())

# b = np.arange(12).reshape(3,4)
# print(b)
# #axis = 0 按列 1 按行
# print(b.sum(axis=0)) 
# print(b.sum(axis=1)) 

# #通用函数
# print(np.exp(b))
# print(np.sqrt(b))

#索引、切片、迭代

# a = np.arange(10)
# b = a[:6:] #numpy的切片生成的是视图 不会产生新的数组 不会占用新的内存
# print(a)
# b[0] = 999
# print(a) #a[0]=999
# a[:8:2] = 1000 #左边切片选中 0 2 4 6 右边赋值 广播
# print(a)
# a[:8:2] = [12,23,34,45] #分别赋值 数量不对应会报错
# print(a)
# a = a[::-1]
# print(a)
# for i in a:
#     print(i**2)
# def f(x,y):
#     return x*10 + y
# b = np.fromfunction(f,(5,4),dtype = np.int64) #根据函数生成一个数组 函数 shape dtype
#                                             #先生成行坐标x 再生成列坐标y 大小都是5*4 
# print(b)
# print(b[2,3]) #三行四列元素
# print(b[:,2]) #所有行 第3列
# print(b[1:3,1:3])#第2行第3行 第2列第3列
# print(b[-1])#-1代表最后1行/列 没有索引 默认完整 b[-1]就代表最后一行所有元素

# c = np.array([[[1,2,3],[2,3,4],[3,4,1],[5,6,9]],
#               [[112,342,6],[213,24,124],[12,21,12],[123,431,241]]])
# c1 = np.array([[[1,2,3],[2,3,4]],
#                [[3,4,1],[5,6,9]],
#                [[112,342,6],[213,24,124]],
#                [[12,21,12],[123,431,241]]])
# print(c)
# print(c.shape)
# print(c1.shape)
# print(c[1,...])#c[1,...] = c[1,:,:] = c[1] 第二行
# print(c[...,1])#c[...:1] = c[:,:,1] 矩阵中每一个数组元素的第二个元素
# print(c[:,2:4,0:2])#矩阵每一行 第三列-第四列 第一个第二个元素

# #迭代 
# for row in c: #对每一行迭代
#     print(row)
# for element in c.flat: #对每一个元素迭代 .flat
#     print(element)

# #形状操作
# rg = np.random.default_rng(5)
# a = np.floor(10*rg.random((3,4))) #floor 不大于x的最大整数 random默认从0-1 不*10 a显示的全是0.
# print(a) #生成0-10不包括10 的随机整数3*4矩阵
# a_ravel = a.ravel() #降成1维 a不变 
#                     #C-style 平铺时，右边的索引变化快 00 01 02 03 10 11 12 13 列变化4个 行变化1个  
#                     #FORTRAN-style 正好相反
# print(a_ravel)
# a_reshape = a.reshape(2,6)#不修改原数组
# print(a_reshape)
# a_resize = a.resize((1,12))#传tuple
# print(a)#resize 修改原数组
# print(a_resize)#没有返回值 原地修改数组
# a_T = a.T
# print(a_T)
# b = a.reshape(4,-1) #给出一个维度 另一个填-1 就会自动计算
# print(b)
rg = np.random.default_rng(10)
#堆叠
# a = np.floor(10*rg.random((2,2)))
# b = np.floor(10*rg.random((2,2)))

# c = np.vstack((a,b)) #垂直堆叠
# d = np.hstack((a,b)) #水平堆叠

# print(c,d)

# e = np.column_stack((a,b))
# print(e)

a = np.floor(10*rg.random(4))
b = np.floor(10*rg.random(4))

c = np.vstack((a,b)) #垂直堆叠
d = np.hstack((a,b)) #水平堆叠

print(c,d)

e = np.column_stack((a,b)) #针对一维数组 把一维数组当作一列来处理 虽然也是“水平”排列                  
print(e)                   #但结果与hstack不同

f = np.hstack((a[:,newaxis],b[:,newaxis])) # 变成列向量(2维)结果和colnmu_stack相同
                                           # 先把a,b增加一个列向量 然后水平拼接
print(f)
#newaxis 可用于将一维数组变成行向量或列向量
a = np.array([1,2,3,4,5])
print(a.shape)
print(a)
# a = a[:,np.newaxis]
# print(a.shape)
# print(a)

a = a[np.newaxis,:]
print(a.shape)
print(a)


