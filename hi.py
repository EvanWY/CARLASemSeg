import cv2
import numpy as np
import sys

img = cv2.imread('/Users/yang/Desktop/img0001790.png')
img2 = cv2.imread('/Users/yang/Desktop/seg0001790.png')

cv2.imshow('image1', img)
cv2.imshow('image2', img2 * 20)


cv2.waitKey(0)
cv2.destroyAllWindows()