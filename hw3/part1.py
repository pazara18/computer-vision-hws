import cv2
import numpy as np
from scipy import io
import os

test_ground_truth_dir = 'BSR/BSDS500/data/groundTruth/test/'
test_images_dir = 'BSR/BSDS500/data/images/test/'

ground_truths = os.listdir(test_ground_truth_dir)
images = os.listdir(test_images_dir)

precision_list = []
for i in range(200):
    file = io.loadmat(test_ground_truth_dir + ground_truths[i])
    image = np.zeros(file['groundTruth'][0][0][0][0][1].shape, dtype=np.uint8)
    for j in range(file['groundTruth'].shape[1]):
        image = cv2.add(image, (file['groundTruth'][0][j][0][0][1] * 255).astype(np.uint8))

    canny = cv2.Canny(cv2.imread(test_images_dir + images[i]), 100, 200)
    tp = np.count_nonzero((canny == 255) & (image == 255))
    fp = np.count_nonzero((canny == 255) & (image == 0))
    precision_list.append(tp / (tp + fp))

avg = np.mean(precision_list)
print(avg)
