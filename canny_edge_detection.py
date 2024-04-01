import cv2
import numpy as np
import matplotlib.pyplot as plt
# Function for performing Canny edge detection
def canny_edge_detection(img_path):
    # Read the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform Canny edge detection with thresholds 100 and 200
    edges = cv2.Canny(img, 100, 200)
    
    # Invert the edges to get a black background with white edges
    inverted_edges = cv2.bitwise_not(edges)
    
    # Invert the image back to the original state (white background with black edges)
    canny = cv2.bitwise_not(inverted_edges)
    
    # Return the Canny edge-detected image
    return canny