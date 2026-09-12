import tifffile
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import time
import argparse

img = tifffile.imread("Data/learning_picture.tif")
# tifffile.imread(path)
# 作用：读取 tif / tiff 文件
# 参数 path：图像文件的路径
# 返回值通常是一个 numpy.ndarray

print(type(img))
# 查看 img 的数据类型
# <class 'numpy.ndarray'>

print(img.shape)
# shape：查看数组的形状
# (128,128,1)
# 高度 × 宽度 × 通道数

print(img.dtype)
# dtype：查看数组中每个像素的数据类型
# float32

img2d = img[:, :, 0]
# 数组切片：取出2D图像 所有行 所有列 第0个通道

print(img2d.shape)
# (128,128)

print("min:", img.min())
print("max:", img.max())
print("mean:", img.mean())

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(10, 4)
)

# plt.subplots()
# 作用：创建一个 Figure，并在里面创建子图
# 1, 2：1 行 2 列
# figsize=(10, 4)：整个画布大小 宽 10 英寸，高 4 英寸
# fig：整个画布
# ax1、ax2：两个子图对象

im = ax1.imshow(
    img2d,
    cmap="gray"
)
# imshow() 作用：显示二维图像
# img2d：要显示的数据
# cmap="gray"：使用灰度颜色映射

fig.colorbar(
    im,
    ax=ax1
)
# colorbar() 作用：添加颜色条

# im：根据哪一张图的数值生成颜色条
#
# ax=ax1：把颜色条绑定到 ax1 子图

ax1.set_title("image")
# set_title()：设置子图标题

ax2.hist(
    img2d.ravel(),
    bins=128
)
# hist() 作用：绘制直方图
# img2d.ravel()：把二维数组变成一维数组
# bins=128：把像素值范围分成 128 个区间进行统计

ax2.set_title("histogram")

plt.tight_layout()
# tight_layout() 自动调整子图之间的间距 防止标题、坐标轴等内容发生重叠

filtered = ndimage.gaussian_filter(
    img2d,
    sigma=1
)
# gaussian_filter() 作用：对图像进行高斯滤波 / 高斯模糊
# img2d：输入图像
# sigma：高斯核的标准差 可以理解为“模糊程度”

fig, axes = plt.subplots(1, 2)
# 创建：1 行 2 列的子图
# axes 是包含两个子图对象的数组
# axes[0] -> 第一个子图
# axes[1] -> 第二个子图

axes[0].imshow(
    img2d,
    cmap="gray"
)

axes[0].set_title("Original")
# 显示原图

axes[1].imshow(
    filtered,
    cmap="gray"
)
axes[1].set_title("Gaussian filter")
# 显示高斯滤波后的图像

# plt.show()
# show()：把前面创建的图显示出来

parser = argparse.ArgumentParser()
# ArgumentParser()
# 创建一个命令行参数解析器parser来让程序可以接收这样的参数：
# python test.py --delay 0.1 --count 100 --desc "Loading"

parser.add_argument(
    "--delay",
    type=float,
    default=0.05
)
# add_argument()
# 作用：定义一个可以从命令行传入的参数
# "--delay"：参数名字
# type=float：参数必须转换成浮点数
# default=0.05：如果用户没有输入 --delay 就默认使用 0.05

parser.add_argument(
    "--count",
    type=int,
    default=200
)

parser.add_argument(
    "--desc",
    type=str,
    default="Processing"
)

args = parser.parse_args()
# parse_args()
# 读取用户在命令行中输入的参数 之后通过：args.delay args.count args.desc来获取对应参数

for i in tqdm(range(args.count),desc=args.desc):
    time.sleep(args.delay)
# range(args.count) 循环 args.count 次
# tqdm(...) 给这个循环添加进度条
# desc=args.desc 设置进度条前面的文字
#time.sleep(args.delay) 每一次循环暂停一段时间

