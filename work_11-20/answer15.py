'''
Soble(索贝尔) 滤波器
    soble滤波器可以提起特定方向的边缘：横向或者纵向

	Sobel算子在Prewitt算子的基础上增加了权重的概念，
	认为相邻点的距离远近对当前像素点的影响是不同的，
	距离越近的像素点对应当前像素的影响越大，从而实现图像锐化并突出边缘轮廓。
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


# sobel filter
def sobel_filter(img, K_size=3):
	if len(img.shape) == 3:
		H, W, C = img.shape
	else:
		# img = np.expand_dims(img, axis=-1)
		H, W = img.shape

	# Zero padding
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
	tmp = out.copy()

	out_v = out.copy()
	out_h = out.copy()

	## 垂直方向
	Kv = [[1., 2., 1.],
		  [0., 0., 0.],
		  [-1., -2., -1.]]
	## 水平方向
	Kh = [[1., 0., -1.],
		  [2., 0., -2.],
		  [1., 0., -1.]]

	# filtering
	for y in range(H):
		for x in range(W):
			out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
			out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))

	out_v = np.clip(out_v, 0, 255)
	out_h = np.clip(out_h, 0, 255)

	out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
	out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out_v, out_h,out_h+out_v

# Read image
img = cv2.imread("example.jpg").astype(np.float)

# grayscale
gray = BGR2GRAY(img)

# sobel filtering
out_v, out_h,out = sobel_filter(gray, K_size=3)

# Save result
cv2.imwrite("out_v15.jpg", out_v)

cv2.imwrite("out_h15.jpg", out_h)
cv2.imwrite("out_d15.jpg", out)
