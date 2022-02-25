import os
from cv2 import cv2
import moviepy.editor as mpy
import numpy as np

FOLDER_NAME = 'DJI_0101'

frame_paths = os.listdir(FOLDER_NAME)

frames = []
for path in frame_paths:
    img = cv2.imread(FOLDER_NAME + '/' + path, 0)
    frames.append(img)

diff_list = []
for i in range(1, len(frames)):
    diff = cv2.absdiff(frames[i], frames[i - 1])
    ret, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8))
    diff_list.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))

clip = mpy.ImageSequenceClip(diff_list, fps=25)
clip.write_videofile("part-1.2.mp4", codec="libx264")