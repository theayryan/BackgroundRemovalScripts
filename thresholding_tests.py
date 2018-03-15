import cv2
import numpy as np
from matplotlib import pyplot as plt

#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 =200
CANNY_THRESH_2 = 250
MASK_DILATE_ITER = 15
MASK_ERODE_ITER = 15
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


#== Processing =======================================================================

n = input("enter file number: ")
#-- Read image -----------------------------------------------------------------------
img = cv2.imread(('samples/tv'+n+'.jpg'))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection -------------------------------------------------------------------
#edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
#edges = cv2.dilate(edges, None)
#edges = cv2.erode(edges, None)

x, edges = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()