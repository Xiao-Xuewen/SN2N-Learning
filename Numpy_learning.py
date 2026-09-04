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
# # a = np.floor(10*rg.random((2,2)))
# # b = np.floor(10*rg.random((2,2)))

# # c = np.vstack((a,b)) #垂直堆叠
# # d = np.hstack((a,b)) #水平堆叠

# # print(c,d)

# # e = np.column_stack((a,b))
# # print(e)

# a = np.floor(10*rg.random(4))
# b = np.floor(10*rg.random(4))

# c = np.vstack((a,b)) #垂直堆叠
# d = np.hstack((a,b)) #水平堆叠

# print(c,d)

# e = np.column_stack((a,b)) #针对一维数组 把一维数组当作一列来处理 虽然也是“水平”排列                  
# print(e)                   #但结果与hstack不同

# f = np.hstack((a[:,newaxis],b[:,newaxis])) # 变成列向量(2维)结果和colnmu_stack相同
#                                            # 先把a,b增加一个列向量 然后水平拼接
# print(f)
# #newaxis 可用于将一维数组变成行向量或列向量
# a = np.array([1,2,3,4,5])
# print(a.shape)
# print(a)
# # a = a[:,np.newaxis]
# # print(a.shape)
# # print(a)

# a = a[np.newaxis,:]
# print(a.shape)
# print(a)

# a = np.floor(10 * rg.random((2,12)))

# b = np.hsplit(a,3) # 按水平方向 竖着切

# c = np.vsplit(a,2) # 按垂直方向 横着切

# print(a,b,c,sep = "\n")

# #copies and views 
# #两个变量内存共用？不共用？
# #一般的数组 变量名 a --> ndarray 对象 --> 内存
# #1.no copy at all
# #没有创建新的数组b b and a point the same array
# a = np.floor(10 * rg.random((3,4)))
# b = a 
# print(b is a) # is 不是判断数值的 是判断是不是同一个对象

# #2. view 对象不同 但共用内存数据 以不同的视角看待同样的数据 比如不同的shape 切片也是一样的道理
# # a - ndarrayA - 内存A
# # c - ndarrayC - 内存A
# # c = a.view()
# # print(a is c)
# c = a.reshape((2,6))
# print(a,c)
# c[1,2] = 100
# print(a,c) #a和c都变了
# d = a[:,2:]
# d[2,1] = 99
# print(a,d) #a和d都变了

# #注意
# #d[:] = 10 是将d所指的数组元素都变成10 
# #d = 10 是将d重新指向 10 这个变量 a不变 
# #操作数组需要使用数组的规范 否则可能换指变量

# #3.deep copy 真正的复制

# e = a.copy()
# # a - ndarrayA - 内存A
# # d - ndarrayD - 内存D
# #两份数据彻底独立
# #当有一份巨大的数据量时 比如10000个数据 我们只要前100 我们如果只是切片b = a[:100] 
# #然后删除 a 实际上并没有释放a的10000的数据内存 因为b的视图还在指向
# #我们需要 b = a[:100].copy() 之后 del a 才会释放10000的数据的内存

#广播的两个条件满足一个即可：1）两个数组的shape右对齐 2）如果对齐同 其中有1也行
#(4,3) (3,)可以 (4,3)(4,)不行 (15,3,5)(3,5)可以 缺失的维度当作1
#img.shape = (256,256,3) scale.shape = (3,) scale=([1.0,0.8,0.5]) 
#img*scale = img的第三轴与scale对应元素相乘 第一轴和第二轴不变

a = np.array([0,10,20,30])
print(a.shape)

a = a[:,np.newaxis] #(4,1)
print(a.shape)

b = np.array([1,2,3])#(3,)-->"(1,3)" 从右往左边比 两项都不同 但是都有1 结果维度取非1维度-->(4,3)

c = a + b

print(c)
print(c.shape)
#广播典型的用途 批量运算 shape = (4,2) 和 (2,)计算 左边四个2元素向量和基准2元素向量的差 
#直接做差即可 得到shape = (4,2) 每一行元素就是对应的差值 不需要for

#img.shape = (100,512,512) 100张图像 每一个的像素都是512*512

#以显微成像的例子来说 (100,512,512) 理解为 100张图 每张图的高和宽都是 512*512的 
#然后每张图的均值mean计算出来 那就是100个数值 shape = (100,)
#然后直接减法不行 因为最右边100和512对不上 需要增加两个维度shape =(100,1,1)
#使用np.newaxisok或者[:,none,none] 变成(100,1,1) 然后做差 
#现在mean是个三维数组 我的理解就是行和列都是 1 然后垂直于屏幕的轴有100 
#然后原来的图像行和列是 512*512 垂直于屏幕的轴也是100 
#实际的减法就是原图像数组的第n个图像的行和列的每一个点都减去mean数组的对应的第n的图像的mean
#最后的shape 依然是(100,512,512) 此时 这100张图像的每一个图像的每一个像素都减去了对应图像的均值
#得到新的100张512*512的图
#广播的使用条件不是关键 关键是知道广播是如何计算的 然后理解 设计广播
#1)减均值 如上
#2)减同一张图像(512*512)的[none,:,:]做差即可
#3)每一个图像的每一行减去一个对应行数的均值(100,512)-->[:,:,none]-->(100,512,1) 
#shape完全相同的 简单 对应的+ - * / 
#shape不同的 沿着不同的位置轴上做运算 相同的轴不做 这句话有点抽象 
#以3)为例 我们要把每个图像的每一行减去对应的均值 实际上是在沿着列做运算 (a11-am1 a12-am1 a13-am1) 列的序号在增大
#所以我们none的位置是列的位置 而不是[100,1,512]虽然这样也能广播 但这样的广播就不对了
#这样做差的话实际上是沿着行运算 把每一列的元素减去对应的每一行的均值 
#听着有些绕 实际上是因为这是一个正方形例子 刚好这么设计也能计算
#如果这是一个h w不相等的 比如(100,500,700)初始图像
#取每一个图像的每一行的均值(100,500) 怎么变 一定是[:,:,none]因为如果中间none 广播不了 
#但根据这种设计的话 是根据广播规则来设计的 不是根据计算原理来设计的
#例子2) 同时减去同一张图像 是不是就沿着垂直于电脑的轴进行运算的 所以[none,:,:]
