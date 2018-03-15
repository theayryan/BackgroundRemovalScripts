import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def get_holes(image, thresh):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    im_bw = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]
    #im_bw = cv.adaptiveThreshold(gray, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)[1]
    #cv.imshow('img', im_bw)
    # Display
    #cv.waitKey()
    im_bw_inv = cv.bitwise_not(im_bw)
    #cv.imshow('img', im_bw_inv)
    # Display
    #cv.waitKey()
    _,contour,_ = cv.findContours(im_bw_inv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #for cnt in contour:
    #    cv.drawContours(im_bw_inv, [cnt], 0, 255, -1)
    cv.drawContours(im_bw_inv, contour, -1,  (255,0,0), cv.FILLED)
    #cv.imshow('img', im_bw_inv)
    # Display
    #cv.waitKey()
    nt = cv.bitwise_not(im_bw)
    im_bw_inv = cv.bitwise_or(im_bw_inv, nt)
    return im_bw_inv


def remove_background(image, thresh, scale_factor=.70, kernel_range=range(2), border=0):
    border = border or kernel_range[-1]

    holes = get_holes(image, thresh)
    small = cv.resize(holes, None, fx=scale_factor, fy=scale_factor)
    bordered = cv.copyMakeBorder(small, border, border, border, border, cv.BORDER_CONSTANT, 0)

    for i in kernel_range:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*i+1, 2*i+1))
        bordered = cv.morphologyEx(bordered, cv.MORPH_CLOSE, kernel)

    unbordered = bordered[border: -border, border: -border]
    mask = cv.resize(unbordered, (image.shape[1], image.shape[0]))
    fg = cv.bitwise_and(image, image, mask=mask)
    return fg


img = cv.imread('samples/tv6.jpg')
nb_img = remove_background(img, 230)

nb11 = remove_background(nb_img, 230)

cv.imshow('img', nb11)
# Display
cv.waitKey()