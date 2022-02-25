import cv2
import numpy as np
import os
from scipy import io

output_path = 'CATS-master/output/bsds/test/sing_scale_test/'
ground_truth_dir = 'BSR/BSDS500/data/groundTruth/test/'

ground_truths = os.listdir(ground_truth_dir)
outputs = os.listdir(output_path)
T = 127

precision_list = []
for i in range(200):
    file = io.loadmat(ground_truth_dir + ground_truths[i])
    image = np.zeros(file['groundTruth'][0][0][0][0][1].shape, dtype=np.uint8)
    for j in range(file['groundTruth'].shape[1]):
        image = cv2.add(image, (file['groundTruth'][0][j][0][0][1] * 255).astype(np.uint8))
        
    output = cv2.cvtColor(cv2.imread(output_path + outputs[i]), cv2.COLOR_RGB2GRAY)
    T, output = cv2.threshold(output, T, 255, cv2.THRESH_BINARY)
    tp = np.count_nonzero((image == 255) & (output == 255))
    fp = np.count_nonzero((image == 255) & (output == 0))
    precision_list.append(tp / (tp + fp))

avg = np.mean(precision_list)
print(avg)
