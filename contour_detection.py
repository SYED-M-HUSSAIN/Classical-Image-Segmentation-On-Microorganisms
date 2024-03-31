import cv2
import numpy as np
import matplotlib.pyplot as plt
def contour_detection(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    canvas = 255 * np.ones_like(img)
    cv2.drawContours(canvas, contours, -1, (0, 0, 0), 3)
    contour = cv2.bitwise_not(canvas)
    return contour