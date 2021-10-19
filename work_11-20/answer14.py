'''
微分滤波器
    微分滤波器对图像亮度急剧变化的边缘有提取效果，可以获得邻近像素的差值

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
#different filter
def different_filter(img,K_size=3):
	H,W = img.shape

	#Zero padding
	pad = K_size//2
	out = np.zeros((H+pad*2,W+pad*2),dtype=np.float)
	out[pad:pad+H,pad:pad+W] = img.copy().astype(np.float)
	tmp = out.copy()

	out_v = out.copy()
	out_h = out.copy()
	out_d = out.copy()

	# 垂直方向
	Kv = [[0., -1., 0.],
		  [0., 1., 0.],
		  [0., 0., 0.]]
	# 水平方向
	Kh = [[0., 0., 0.],
		  [-1., 1., 0.],
		  [0., 0., 0.]]
	# 对角线方向
	Dh = [[-1., 0., 0.],
		  [0., 1., 0.],
		  [0., 0., 0.]]

	# filtering
	for y in range(H):
		for x in range(W):
			out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
			out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))
			out_d[pad + y, pad + x] = np.sum(Dh * (tmp[y: y + K_size, x: x + K_size]))

	out_v = np.clip(out_v, 0, 255)
	out_h = np.clip(out_h, 0, 255)
	out_d = np.clip(out_d, 0, 255)

	'''
	np.clip(a,a_min,a_max,out=None]
	clip这个函数将将数组中的元素限制在a_min, a_max之间，
	大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
	'''

	out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
	out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)
	out_d = out_d[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out_v, out_h ,out_d,out_h+out_v
# Read image
img = cv2.imread("example.jpg").astype(np.float)

# grayscale
gray = BGR2GRAY(img)
# different filtering
out_v, out_h ,out_d,out= different_filter(gray, K_size=3)

cv2.imwrite("out14_v.jpg", out_v)
cv2.imwrite("out14_h.jpg", out_h)
cv2.imwrite("out14_d.jpg", out_d)
cv2.imwrite("out14.jpg", out_d)