"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Desmond Kelleher (u7033968)
"""

import numpy as np
from skimage import util, color
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.
    return result

def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def window(image, size, a, b):
    total = 0
    #this function takes the image and the current co-ordinates, and returns
    #  the sum of the pixel (a,b) and its surrounding pixels
    #the number of pixels in each direction are given by the size variable
    for i in range(-size, size):
        for j in range(-size, size):
            total += image[a + i, b + j]
    return total

def compare_neighbour(corner, neighbours):
    #'padded_corner' is the 2d array of cornerness values but with 0 padding at its edges
    #the width of the zero padding is given by 'neighbours'
    padded_corner = np.zeros((corner.shape[0] + neighbours, corner.shape[1] + neighbours))
    padded_corner[neighbours:corner.shape[0] + neighbours, neighbours:corner.shape[1] + neighbours] = corner
    #iterate through all entries in 'padded_corner' that are not zero padding
    for i in range(neighbours, image.shape[0]):
        for j in range(neighbours, image.shape[1]):
            #check to see if current entry is greater than surroundings.
            #  Surroundings extend as far as variable 'neighbours'
            #if pixel in neighbourhood is greater than the current pixel, the current pixel is set to 0
            for a in range(-neighbours, neighbours):
                for b in range(-neighbours, neighbours):
                    if(padded_corner[i,j] != 0):
                        if(padded_corner[i,j] < padded_corner[i + a, j + b]):
                            padded_corner[i,j] = 0
                       
    corner = padded_corner[neighbours:corner.shape[0] + neighbours, neighbours:corner.shape[1] + neighbours]
    return corner

def local_maxima(corner, original, div):
    #this function divides the cornerness array into sections ('square'), finds 
    # the greatest entry in each square and sets 
    #all other values to zero. A coloured square is placed into the original 
    # image to indicate the location of a 
    #detected corner
    #'div' gives the width of the sections each axis is divided into
    #the + 1 is added to the for loop as the division of the shape of the 
    # image by 'div' will most likely produce a remainder
    corner_points = []
    for i in range(0, image.shape[0]//div + 1):
        #for the case where there is no remainder
        if(i*div == image.shape[0]):
                break
        for j in range(0, image.shape[1]//div + 1):
            #the upper limit of the x and y coordinates of the section
            y_edge = i*div + div
            x_edge = j*div + div
            
            #for the case where there is no remainder
            if(j*div == image.shape[1]):
                break

            #the x and y limits are redefined for sections at the end of each axis
            #where they would otherwise be out of bounds
            if ((i*div + div) > image.shape[0]):
                y_edge = image.shape[0]
            if ((j*div + div) > image.shape[1]):
                x_edge = image.shape[1]
            #a subarray is created out of the section currently being examined
            square = corner[i*div : y_edge, j*div : x_edge]
            #return the coordinates of the corner of greatest cornerness
            a, b = np.unravel_index(square.argmax(), square.shape)
            #the following code is executed iff the region has a non-zero maximum
            if(square[a,b] != 0):
                #all of the subarray is set to zero, except for the max which is set to
                #255
                square = np.zeros(square.shape)
                square[a,b] = 255
                #the section is reinserted into the cornerness array
                corner[i*div : y_edge, j*div : x_edge] = square
                #the location of the pixel (a,b) has a coloured square inserted into it
                #  in the original image
                colour = (255, 255, 100) * np.ones((2,2,3))
                y = i*div + a
                x = j*div + b
                if(original.shape[0] - 1 > y > 0 and original.shape[1] > x > 0):
                    original[y - 1: y + 1 , x - 1: x + 1] = colour
                    corner_points.append((y, x))
    return corner, original, corner_points

def harris(image, original, Iy2, Ix2, Ixy, neighbours, div, k):
    
    #############################################################
    # Task: Compute the Harris Cornerness
    #############################################################

    M = np.zeros((2, 2))
    corner = np.zeros(image.shape)
    size = 1
    for i in range(size, image.shape[0] - size):
        for j in range(size, image.shape[1] - size):
                #the matrix M is redefined at each pixel 
                M[0,0] = window(Ix2, size, i, j)
                M[1,0] = window(Ixy, size, i, j)
                M[0, 1] = M[1, 0]
                M[1, 1] = window(Iy2, size, i, j)
                #the eigenvalues of M are calculated
                eig, vectors = np.linalg.eig(M)
                #the cornerness at this pixel is caluclated and stored in the cornerness array
                corner[i,j] = eig[0]*eig[1] - k*(eig[0] + eig[1])**2
                #thresholding 
                if (corner[i,j] < thresh):
                    corner[i,j] = 0

    ############################################################
    # Task: Perform non-maximum suppression and
    #       thresholding, return the N corner points
    #       as an Nx2 matrix of x and y coordinates
    ############################################################

    #using builtin function to find cornerness with thresholding
    ############################################################
    builtin = cv2.cornerHarris(image, neighbours, 3, k)
    for i in range(0,image.shape[0]):
        for j in range(0, image.shape[1]):
            if(builtin[i,j] <= 1e-4):
                builtin[i, j] = 0
    #############################################################

    #non-minimum suppression on both sets of corners by comparing individual pixels
    #with neighbours and taking maxima of sections
    #############################################################
    builtin = compare_neighbour(builtin, neighbours)
    corner = compare_neighbour(corner, neighbours)

    corner, original_a, corner_list = local_maxima(corner, original, div)
    builtin, original_b, corner_list_b = local_maxima(builtin, original, div)
    ##############################################################

    #returning corner points as an array 
    corner_points = np.array(corner_list)

    #show images
    cv2.imshow('cv2.cornerHarris', original_b)
    cv2.imshow('User-defined', original_a)
    return corner_points
    


# Parameters, add more if needed
sigma = 2
thresh = 1*10**8

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import matplotlib.pyplot as plt


#input image
image = cv2.imread('Harris_3.jpg')

image = np.uint8(image)
original = image
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# computer x and y derivatives of image
Ix = conv2(image, dx)
Iy = conv2(image, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

################################################################
#################Harris Corner Detection########################
neighbours = 6
div = 12
k = 0.02

points = harris(image, original, Iy2, Ix2, Ixy, neighbours, div, k)
cv2.waitKey(0)

################################################################

