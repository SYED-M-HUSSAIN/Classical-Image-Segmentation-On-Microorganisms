import os
import cv2
import matplotlib.pyplot as plt
# from fast_marching_segmentation import fast_marching_segmentation
from threshold_segmentation import threshold_segmentation
from canny_edge_detection import canny_edge_detection
from contour_detection import contour_detection
from kmeans_segmentation import kmeans_segmentation
from watershed_segmentation import watershed_segmentation

# Paths to your images
img_paths = [
    '/home/hussain/Classical Image Segmentation/test_data/EMDS5-g01-40.png','/home/hussain/Classical Image Segmentation/test_data/EMDS5-g02-21.png'
]

import os

# Create a directory to save the results
output_dir = 'segmentation_results'
os.makedirs(output_dir, exist_ok=True)

# Perform segmentation
segmentation_functions = [
    ("Threshold Segmentation", threshold_segmentation),
    ("Canny Edge Detection", canny_edge_detection),
    ("Contour Detection", contour_detection),
    ("K-means Segmentation", kmeans_segmentation),
    ("Watershed Segmentation", watershed_segmentation)
]

results = []
for img_path in img_paths:
    img_name = img_path.split('/')[-1].split('.')[0]
    img_results = [(seg_name, segmentation_func(img_path)) for seg_name, segmentation_func in segmentation_functions]
    results.append((img_name, img_results))

# Create a single image with sub-images
num_imgs = len(img_paths)
num_segs = len(segmentation_functions)

fig, axes = plt.subplots(num_imgs, num_segs, figsize=(16, 12))

for i, (img_name, img_results) in enumerate(results):
    for j, (seg_name, result) in enumerate(img_results):
        ax = axes[i, j] if num_imgs > 1 else axes[j]
        ax.imshow(result, cmap='gray')
        ax.axis('off')
        ax.set_title(f"{seg_name}")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'segmentation_results.png'))
plt.show()