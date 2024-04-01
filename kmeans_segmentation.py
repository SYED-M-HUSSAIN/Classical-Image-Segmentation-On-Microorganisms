import cv2
import numpy as np
import matplotlib.pyplot as plt
# Function for performing K-means segmentation on an image
def kmeans_segmentation(img_path):
    # Read the image
    img = cv2.imread(img_path)
    
    # Reshape the image to a 2D array of pixels
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    
    # Define the criteria for the kmeans algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Define the number of clusters (K)
    K = 100
    
    # Apply kmeans algorithm
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert the center values to uint8
    center = np.uint8(center)
    
    # Reshape the result to the original image shape
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    # Convert the segmented image to grayscale
    grayscale_res = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to obtain a binary image
    _, binary_res = cv2.threshold(grayscale_res, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Invert the binary image
    kmean = cv2.bitwise_not(binary_res)
    
    # Return the K-means segmented image
    return kmean