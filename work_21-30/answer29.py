import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
仿射变换-放大缩小
'''

# Affine
def affine(_img, a, b, c, d, tx, ty):
	H, W, C = _img.shape

	# temporary image
	img = np.zeros((H + 2, W + 2, C), dtype=np.float32)
	img[1:H + 1, 1:W + 1] = _img

	# get new image shape
	H_new = np.round(H * d).astype(np.int)
	W_new = np.round(W * a).astype(np.int)
	out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

	# get position of new image
	x_new = np.tile(np.arange(W_new), (H_new, 1))
	y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

	# get position of original image by affine
	adbc = a * d - b * c
	x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
	y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

	x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
	y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

	# assgin pixcel to new image
	out[y_new, x_new] = img[y, x]

	out = out[:H_new, :W_new]
	out = out.astype(np.uint8)

	return out


# Read image
img = cv2.imread("example.jpg").astype(np.float32)

# Affine
out = affine(img, a=1.3, b=0, c=0, d=0.8, tx=0, ty=0)


# Save result

cv2.imwrite("out29.jpg", out)