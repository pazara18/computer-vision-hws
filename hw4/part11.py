import os
from cv2 import cv2
import numpy as np
import moviepy.editor as mpy

FOLDER_NAME = 'DJI_0101'

frame_paths = os.listdir(FOLDER_NAME)

frames = []
for path in frame_paths:
    img = cv2.imread(FOLDER_NAME + '/' + path, 0)
    frames.append(img)

# https://stackoverflow.com/questions/17931613/how-to-decide-a-whether-a-matrix-is-singular-in-python-numpy
def is_invertible(M):
    return M.shape[0] == M.shape[1] and np.linalg.matrix_rank(M) == M.shape[0]


h, w = frames[0].shape

window_size = 5

smooth_kernel = np.ones((5, 5), np.float32) / 25
video_frames = []

dy1, dx1 = np.gradient(frames[0])
current_image_smooth = cv2.blur(frames[0], (5, 5))
prev_image_smooth = current_image_smooth
dy2, dx2 = dy1, dx1
for i in range(1, len(frames)):
    current_image = frames[i]
    arrowed_image = cv2.imread(FOLDER_NAME + '/' + frame_paths[i])
    corner = cv2.goodFeaturesToTrack(current_image, 100, 0.01, 10)
    corners = np.int0(corner)
    dy1, dx1 = np.gradient(current_image)
    Ix = (dx1 + dx2) / 2
    Iy = (dy1 + dy2) / 2
    current_image_smooth = cv2.blur(current_image, (5, 5))
    It = cv2.absdiff(current_image_smooth, prev_image_smooth)
    ret, It = cv2.threshold(It, 5, 255, cv2.THRESH_BINARY)
    for corner in corners:
        y, x = corner.ravel()

        AtA = np.zeros((2, 2))
        AtB = np.zeros((2, 1))
        for window_x in range(-(window_size - 1), window_size):
            for window_y in range(-(window_size - 1), window_size):
                if x + window_x >= current_image.shape[0] or y + window_y >= current_image.shape[1]:
                    Ix_val = 0
                    Iy_val = 0
                    It_val = 0
                elif x - window_x <= -1 or y - window_y <= -1:
                    Ix_val = 0
                    Iy_val = 0
                    It_val = 0
                else:
                    Ix_val = Ix[x + window_x, y + window_y]
                    Iy_val = Iy[x + window_x, y + window_y]
                    It_val = It[x + window_x, y + window_y]

                AtA += [[Ix_val*Ix_val, Ix_val*Iy_val], [Iy_val*Ix_val, Iy_val*Iy_val]]

                AtB += [[Ix_val*It_val], [Iy_val*It_val]]

        if is_invertible(AtA) and not np.all(AtB == 0):
            motion_field = -(np.linalg.inv(AtA).dot(AtB))
            new_x = int(x + motion_field[0] * 2)
            new_y = int(y + motion_field[1] * 2)

            start = y, x
            end = new_y, new_x

            cv2.arrowedLine(arrowed_image, start, end, color=(0, 255, 255), thickness=1)

    dy2, dx2 = dy1, dx1
    prev_image_smooth = current_image_smooth
    video_frames.append(arrowed_image[:, :, [2, 1, 0]])

clip = mpy.ImageSequenceClip(video_frames, fps=25)
clip.write_videofile("part-1.1.mp4", codec="libx264")
