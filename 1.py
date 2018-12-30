import numpy as np
import cv2 as cv
import math

noble = cv.imread('noble.jpg', cv.IMREAD_COLOR)
lena = cv.imread('lena.jpg', cv.IMREAD_COLOR)
result = cv.imread('noble.jpg', cv.IMREAD_COLOR)

red=lena.copy()
red[:,:,0]=0
red[:,:,1]=0

X = noble.shape[0]
Y = noble.shape[1]
R = math.sqrt((X/2) * (X/2) + (Y/2) * (Y/2))
T = 30
for t in range(0, T + 1):
    r = math.floor(R * t / T)
    for x in range(0, X):
        for y in range(0, Y):
            d = math.sqrt((X/2 - x) * (X/2 - x) + (Y/2 - y) * (Y/2 - y))
            if d < r:
                result.itemset((x, y, 0), red.item(x, y, 0))
                result.itemset((x, y, 1), red.item(x, y, 1))
                result.itemset((x, y, 2), red.item(x, y, 2))
            else:
                result.itemset((x, y, 0), noble.item(x, y, 0))
                result.itemset((x, y, 1), noble.item(x, y, 1))
                result.itemset((x, y, 2), noble.item(x, y, 2))
    cv.imshow('transition', result)
    cv.waitKey(1)
cv.waitKey(0)
cv.destroyAllWindows()
