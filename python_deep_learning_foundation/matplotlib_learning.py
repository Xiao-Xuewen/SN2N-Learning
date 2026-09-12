import numpy as np
import matplotlib.pyplot as plt

img = np.array([
    [0, 50, 100],
    [50, 150, 200],
    [100, 200, 255]
])

im = plt.imshow(img, cmap="gray") # 灰度图
plt.colorbar(im)
plt.show()
