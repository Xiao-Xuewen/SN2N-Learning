import tifffile
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import time
import argparse

img = tifffile.imread("Data/learning_picture.tif")

print(type(img))
print(img.shape)
print(img.dtype)

img2d = img[:, :, 0] #第0个通道也就是唯一的那个通道
print(img2d.shape)

print("min:", img.min())
print("max:", img.max())
print("mean:", img.mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

# 子图1：显示灰度图
im = ax1.imshow(img2d, cmap="gray")
fig.colorbar(im, ax=ax1)   # colorbar绑定给ax1子图
ax1.set_title("image")

# 子图2：灰度直方图
ax2.hist(img2d.ravel(), bins=128)
ax2.set_title("histogram")

plt.tight_layout() # 自动调整间距，防止标题重叠


filtered = ndimage.gaussian_filter(img2d, sigma=1) #sigma 控制高斯滤波“影响周围多大范围”。

fig, axes = plt.subplots(1, 2)

axes[0].imshow(img2d, cmap="gray")
axes[0].set_title("Original")

axes[1].imshow(filtered, cmap="gray")
axes[1].set_title("Gaussian filter")

# plt.show()

parser = argparse.ArgumentParser() #创建一个参数解析器

parser.add_argument(
    "--delay",
    type=float,
    default=0.05
)

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


args = parser.parse_args() # 接收参数,使用时，使用args.参数名 比如args.delay args.count args.desc

for i in tqdm(range(args.count), args.desc):
    time.sleep(args.delay)
