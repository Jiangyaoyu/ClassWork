#最大池化
import cv2
import numpy as np

#max pooling
def max_pooling(img,G=8):
    #max pooling
    out = img.copy()
    H,W,C = img.shape
    Nh = int(H/G)
    Nw= int(W/G)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1),G*x:G*(x+1),c]=np.max(out[G*y:G*(y+1),G*x:G*(x+1),c])
    return out
#read image
img = cv2.imread("imori.jpg")
#max pooling
out = max_pooling(img)

#save result
cv2.imwrite("out8.jpg",out)
