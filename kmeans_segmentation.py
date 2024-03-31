import cv2
import numpy as np
import matplotlib.pyplot as plt
def kmeans_segmentation(img_path):
    img = cv2.imread(img_path)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 100
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    grayscale_res = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    _, binary_res = cv2.threshold(grayscale_res, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kmean = cv2.bitwise_not(binary_res)
    return kmean