#平均池化
import cv2
import numpy as np

#average pooling
def average_pooling(img,G=8):
    out = img.copy()

    H,W,C = img.shape
    Nh = int(H/G)
    Nw = int(W/G)
    print("Nh:{},Nw:{},c:{}".format(Nh,Nw,C))
    for y in range(Nh):#0-15
        for x in range(Nw):#0-15
            for c in range(C):#0-2
                out[G * y:G * (y + 1), G * x:G * (x + 1), c] = np.mean(
                    out[G * y:G * (y + 1), G * x:G * (x + 1), c]).astype(np.int)
                print(np.mean(
                    out[G * y:G * (y + 1), G * x:G * (x + 1), c]).astype(np.int))

    return out

#read image
img = cv2.imread("imori.jpg")

out = average_pooling(img)

#save result
cv2.imwrite("out7.jpg",out)
cv2.imshow("result",out)