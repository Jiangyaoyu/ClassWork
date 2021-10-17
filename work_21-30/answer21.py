'''
直方图归一化（Histogram Normalization）
        https://blog.csdn.net/Ibelievesunshine/article/details/104918524
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


# histogram normalization
def hist_normalization(img, a=0, b=255):
    # get max and min
    c = img.min()
    d = img.max()

    out = img.copy()

    # normalization
    out[out < c] = a
    out[out > d] = b
    out = (b - a) / (d - c) * (out - c) + a
    # out[out < a] = a
    # out[out > b] = b
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori_dark.jpg").astype(np.float)
H, W, C = img.shape

# histogram normalization
out = hist_normalization(img)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_his21.png")
cv2.imwrite("out21.jpg", out)
