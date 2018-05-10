import numpy as np
import cv2
from math import sqrt

def euc_dist(p1, p2):
    b = int(p1[0])-int(p2[0])
    g = int(p1[1])-int(p2[1])
    r = int(p1[2])-int(p2[2])
    return sqrt(b*b + g*g +r*r);

def printMouseCoord(event, y, x, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('X: ' + str(y) + ' Y: ' + str(x) + ' Color: ' + str(img[x][y]))

img_name = './Images/SkinDataset/GT/Train/11.jpg'
img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', printMouseCoord)
cv2.imshow('image', img)

while(True):

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
