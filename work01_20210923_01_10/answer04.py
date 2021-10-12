 #大津算法   大津算法是一种图像二值化算法，作用是确定将图像分成黑白两个部分的阈值
'''
参考：https://blog.csdn.net/Galen_xia/article/details/107911867?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163331325916780262539575%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163331325916780262539575&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-3-107911867.pc_search_result_hbase_insert&utm_term=%E5%A4%A7%E6%B4%A5%E4%BA%8C%E5%80%BC%E5%8C%96python&spm=1018.2226.3001.4187
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

# Otsu Binarization
def otsu_binarization(img, th=128):
	max_sigma = 0
	max_t = 0

	# determine threshold
	for _t in range(1, 255):
		v0 = out[np.where(out < _t)]
		m0 = np.mean(v0) if len(v0) > 0 else 0.
		w0 = len(v0) / (H * W)
		v1 = out[np.where(out >= _t)]
		m1 = np.mean(v1) if len(v1) > 0 else 0.
		w1 = len(v1) / (H * W)
		sigma = w0 * w1 * ((m0 - m1) ** 2)
		if sigma > max_sigma:
			max_sigma = sigma
			max_t = _t

	# Binarization
	print("threshold >>", max_t)
	th = max_t
	out[out < th] = 0
	out[out >= th] = 255

	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C =img.shape


# Grayscale
out = BGR2GRAY(img)

# Otsu's binarization
out = otsu_binarization(out)

# Save result
cv2.imwrite("out4.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()