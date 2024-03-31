import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny_edge_detection(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    inverted_edges = cv2.bitwise_not(edges)
    canny = cv2.bitwise_not(inverted_edges)
    return canny