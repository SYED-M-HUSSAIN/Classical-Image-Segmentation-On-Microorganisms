import cv2
import numpy as np
import matplotlib.pyplot as plt
# Function for performing watershed segmentation on an image
def watershed_segmentation(img_path):
    # Read the image
    img = cv2.imread(img_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to obtain a binary image (inverted)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Define a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    # Dilate the binary image to get the background
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
    
    # Calculate the distance transform
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    
    # Threshold the distance transform to obtain foreground markers
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    
    # Convert the sure_fg to uint8
    sure_fg = sure_fg.astype(np.uint8)
    
    # Subtract sure_fg from sure_bg to get the unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Return the unknown region (potential objects for watershed segmentation)
    return unknown
