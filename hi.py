import cv2
import numpy as np

img = cv2.imread('/Users/yang/Work/CARLASemSeg/Train/CameraSeg/14.png')
img2 = cv2.imread('/Users/yang/Desktop/test_sem.png')

cv2.imshow('image1',img * 20)
cv2.imshow('image2', img2 * 20)


cv2.waitKey(0)
cv2.destroyAllWindows()
