#Laplacian滤波器
'''

'''
import cv2
import numpy as np


# Gray scale
def BGR2GRAY(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	# Gray scale
	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
	out = out.astype(np.uint8)

	return out

# laplacian filter
def laplacian_filter(img, K_size=3):
	H, W = img.shape

	# zero padding
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
	out2 = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
	tmp = out.copy()

	# laplacian kernle
	K = [[0., 1., 0.],
		 [1., -4., 1.],
		 [0., 1., 0.]]

	K2 = [[1., 1., 1.],
		 [1., -8., 1.],
		 [1., 1., 1.]]

	# filtering
	for y in range(H):
		for x in range(W):
			out[pad + y, pad + x] = np.sum(K * (tmp[y: y + K_size, x: x + K_size]))
			out2[pad + y, pad + x] = np.sum(K2 * (tmp[y: y + K_size, x: x + K_size]))

	out = np.clip(out, 0, 255)
	out2 = np.clip(out2, 0, 255)
	out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
	out2 = out2[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out,out2

# Read image
img = cv2.imread("example.jpg").astype(np.float)

# grayscale
gray = BGR2GRAY(img)

# prewitt filtering
out ,out2= laplacian_filter(gray, K_size=3)


# Save result
cv2.imwrite("out17_1.jpg", out)
cv2.imwrite("out17_2.jpg", out2)