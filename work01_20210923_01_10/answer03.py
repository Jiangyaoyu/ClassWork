#二值化
import cv2
import numpy as np

def BGR2GRAY(img):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()

    #gray scale
    out = 0.2126* r +0.7152 *g +0.0722 *b
    out = out.astype(np.uint8)
    return out

#二值化
def binarization(img,th=128):
    img[img<th]=0
    img[img>th] = 255
    return img

#read image
img = cv2.imread("imori.jpg",0)

#grayscale
#out = BGR2GRAY(img)
#二值化
out = binarization(img)

cv2.imwrite("out3.jpg",out)


