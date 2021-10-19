'''
MAX-MIN 滤波
    使用网格内像素的最大值核最小值的差值对网格内像素重新赋值
    通常使用过边缘检测，
     我们知道，图像的细节属于低频信息，图像的边缘属于高频信息。
     我们使用一定大小的 Max-Min 滤波器作用于图像，
     当滤波器作用于图像细节时，输出结果往往趋向于0（黑色）
     ；而滤波器作用于图像边缘时，Max-Min 输出结果往往趋向于255（白色）。
     所以 最大-最小滤波器 能有效地用于检测图像的边缘和轮廓。
'''
import cv2
import numpy as np

def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

def max_min_filter(img,K_size=3):
    H,W = img.shape

    pad = K_size//2
    out = np.zeros((H+pad*2,W+pad*2),dtype=np.float)
    out[pad:pad+H,pad:pad+W]=img.copy().astype(np.float)
    tmp = out.copy()
    #filetering
    for y in range(H):
        for x in range(W):
            out[pad+y,pad+x] = np.max(tmp[y:y+K_size,x:x+K_size])-np.min(tmp[y:y+K_size,x:x+K_size])
    out = out[pad:pad+H,pad:pad+W].astype(np.uint8)
    return  out

#读取图片
img = cv2.imread("example.jpg").astype(np.float)
gray = BGR2GRAY(img)
out = max_min_filter(gray)
cv2.imwrite("out13.jpg",out)
