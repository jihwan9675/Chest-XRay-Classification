import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = '../pneumonia_dataset/kaggle/chest_xray/'
path2 = '../preprocessing/kaggle/chest_xray/'
ls = ['train/', 'test/', 'val/']
ls2 = ['NORMAL/', 'PNEUMONIA/']

clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))

for i in ls:
    for j in ls2:
        for k in os.listdir(path+i+j):
            img = cv2.imread(path+i+j+k, 0)
            img = cv2.resize(img, dsize=(600,600))
            cla = clahe.apply(img)
            cv2.imwrite(path2+i+j+k, cla)

# img = cv2.imread('../pneumonia_dataset/kaggle/chest_xray/test/NORMAL/IM-0001-0001.jpeg',0)
# img = cv2.imread(path, 0)

# img = cv2.resize(img, dsize=(600,600))
# equ = cv2.equalizeHist(img)
# 
# cl_img = clahe.apply(equ)
# cla = clahe.apply(img)

# cv2.imshow('s',img/255)
# cv2.waitKey()
# cv2.imshow('s',equ)
# cv2.waitKey()
# cv2.imshow('s',cl_img)
# cv2.waitKey()

# plt.hist(img.flat, bins=100, range=(100, 255))
# plt.show()
# plt.hist(equ.flat, bins=100, range=(100, 255))
# plt.show()
# plt.hist(cla.flat, bins=100, range=(100, 255))
# plt.show()
# plt.hist(cl_img.flat, bins=100, range=(100, 255))
# plt.show()

# cv2.imwrite('original.jpeg', img)
# cv2.imwrite('equalizeHist.jpeg', equ)
# cv2.imwrite('equ_clahe.jpeg', cl_img)
# cv2.imwrite('clahe.jpeg', cla)

# plt.hist(img.flat, bins=100, range=(0, 255))
# plt.hist(img, bins=100, range=(0, 255))
#plt.show()
# plt.hist(equ.flat, bins=100, range=(0, 255))
# plt.hist(equ, bins=100, range=(0, 255))
#plt.show()
