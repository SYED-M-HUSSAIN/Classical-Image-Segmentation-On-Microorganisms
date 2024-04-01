import cv2
import numpy as np
import matplotlib.pyplot as plt
# Function for performing threshold segmentation on an image
def threshold_segmentation(img_path):
    # Read the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply a binary threshold to segment the image
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Invert the binary thresholded image
    inverted_thresh = cv2.bitwise_not(thresh)
    
    # Return the inverted thresholded image
    return inverted_thresh