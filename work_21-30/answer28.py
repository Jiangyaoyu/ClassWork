import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
参考：https://www.jianshu.com/p/1cfb3fac3798
"""

# Affine
def affine(_img, a, b, c, d, tx, ty):
	H, W, C = _img.shape

	# temporary image
	img = np.zeros((H+2, W+2, C), dtype=np.float32)
	img[1:H+1, 1:W+1] = _img

	# get new image shape
	H_new = np.round(H * d).astype(np.int)
	W_new = np.round(W * a).astype(np.int)
	out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

	# get position of new image
	x_new = np.tile(np.arange(W_new), (H_new, 1))
	y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

	# 计算目标图像坐标对应的原图像坐标
	adbc = a * d - b * c
	x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
	y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

	# 对原图像坐标中溢出的坐标进行截断处理（取边界值）
	x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
	y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

	# assgin pixcel to new image
	out[y_new, x_new] = img[y, x]

	out = out[:H_new, :W_new]
	out = out.astype(np.uint8)

	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out = affine(img, a=1, b=0, c=0, d=1, tx=30, ty=-30)

# Save result

cv2.imwrite("out28.jpg", out)