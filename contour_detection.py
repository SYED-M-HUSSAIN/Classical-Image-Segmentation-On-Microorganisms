import cv2
import numpy as np
import matplotlib.pyplot as plt
# Function for detecting contours in an image
def contour_detection(img_path):
    # Read the image
    img = cv2.imread(img_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to convert the image to binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a canvas filled with white color
    canvas = 255 * np.ones_like(img)
    
    # Draw contours on the canvas with black color
    cv2.drawContours(canvas, contours, -1, (0, 0, 0), 3)
    
    # Invert the canvas to get black contours on a white background
    contour = cv2.bitwise_not(canvas)
    
    # Return the image with detected contours
    return contour