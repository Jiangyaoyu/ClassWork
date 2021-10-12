import cv2
import numpy as np


#gray scale
def BGR2GRAY(img):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()

    #gray scale
    out = 0.2126* r +0.7152 *g +0.0722 *b
    out = out.astype(np.uint8) #
    return out
#read img
img = cv2.imread("imori.jpg").astype(np.float)

#grayscale
out = BGR2GRAY(img)

#save result
cv2.imwrite("out.jpg",out)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#以灰度模式加载图片，效果同于以上代码
img = cv2.imread("imori.jpg",0) #以灰度模式加载图片，效果同于以上代码

cv2.imwrite("out1.jpg",img)
'''

