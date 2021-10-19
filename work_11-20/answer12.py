'''
 12Motion Filter
 motion滤波与均值滤波和中值滤波类似，均是采用窗口设计，计算窗口的中心值,
 区别就是中心值的方法不同，motion滤波是仅计算窗口矩阵的主对角线元素的均值，
 其他元素不参与计算，以3*3为例，如下所示。中心值就是(x1+x2+x3)/3，
 滤波后的图像每一个像素值都是由该像素点和邻域的窗口元素按照这种规则计算而来。
'''
import cv2
import numpy as np

def motion_filter(img,K_size):
    H,W,C = img.shape

    # Kernel
    K = np.diag([1] * K_size).astype(np.float)
    K /= K_size

    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out