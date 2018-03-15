import numpy as np
import cv2

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

#Load the Image
img0 = cv2.imread('samples/tv1.jpg')
height, width = img0.shape[:2]

#Create a mask holder
mask = np.zeros(img0.shape[:2],np.uint8)

#Grab Cut the object
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = ((height/2) - 15, (width/2) - 15, 30, 30)
cv2.grabCut(img0,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img1 = img0*mask[:,:,np.newaxis]

background = img0 - img1

#Change all pixels in the background that are not black to white
background[np.where((background > [0,0,0]).all(axis = 2))] =[0,0,0]

#Add the background and the image
final = background + img1

#To be done - Smoothening the edges

cv2.imshow('image', final )

k = cv2.waitKey(0)

if k==27:
    cv2.destroyAllWindows()
