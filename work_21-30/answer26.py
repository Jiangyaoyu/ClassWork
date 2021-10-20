import cv2
import numpy as np
import matplotlib.pyplot as plt


# Bi-Linear interpolation
def bl_interpolate(img, ax=1., ay=1.):
	H, W, C = img.shape
	aH = int(ay * H)
	aW = int(ax * W)
	# get position of resized image
	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))
	# 获取对应原图像坐标
	y = (y / ay)
	x = (x / ax)
	#对x,y向下取整 ,取距离最近的点
	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)
	ix = np.minimum(ix, W-2)#防溢出，因为下标是从0开始，所以W-1,又因为是四个点确定一个点，所以W-2
	iy = np.minimum(iy, H-2)
	# get distance求差值
	dx = x - ix
	dy = y - iy
	dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
	dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)
	# interpolation
	out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float)

# Bilinear interpolation
out = bl_interpolate(img, ax=1.5, ay=1.5)

# Save result
cv2.imwrite("out26.jpg", out)