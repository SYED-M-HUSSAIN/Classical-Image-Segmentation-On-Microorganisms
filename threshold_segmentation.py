import cv2
import numpy as np
import matplotlib.pyplot as plt
def threshold_segmentation(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    inverted_thresh = cv2.bitwise_not(thresh)
    return inverted_thresh