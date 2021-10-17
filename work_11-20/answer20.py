'''
直方图
  直方图显示了不同数值的像素出现的次数

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori_dark.jpg").astype(np.float)

# Display histogram
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out20.png")
'''
plt.hist(x,bins,rwidth,rang):
    x：直方图所要用的数据，必须是一维数组，
    bins：直方图的柱数，默认为10
    range:显示的范围，默认是x轴的范围
    rwidth；柱子的宽度占bins宽的比例

img.ravel():
    降维数：默认降维时行序优先，参数’F‘表示列序优先

'''
