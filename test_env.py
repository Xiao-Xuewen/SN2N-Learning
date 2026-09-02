import sys

import cv2
import matplotlib
import numpy as np
import scipy
import skimage
import tifffile
import torch

print("Python executable:")
print(sys.executable)

print("\nPackage check:")
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(2000, 2000, device="cuda")
    y = torch.randn(2000, 2000, device="cuda")
    z = x @ y

    print("GPU calculation device:", z.device)

print("\nEnvironment OK")

"""
修改代码测试
"""
