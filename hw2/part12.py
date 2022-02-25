import numpy as np
from cv2 import cv2
import moviepy.editor as mpy


# TODO
# Optimize the function using numpy for for loops
# Couldn't figure out the numpy way to implement this
def find_harris_corners(input_img, k, window_size, threshold):
    corner_list = []
    output_img = input_img.copy()
    gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)

    offset = window_size // 2
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset

    Iy, Ix = np.gradient(gray)
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    for y in range(offset, y_range):
        for x in range(offset, x_range):
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1

            windowIxx = Ixx[start_y: end_y, start_x: end_x]
            windowIxy = Ixy[start_y: end_y, start_x: end_x]
            windowIyy = Iyy[start_y: end_y, start_x: end_x]

            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            det = (Sxx * Syy) - (Sxy * Sxy)
            trace = Sxx + Syy

            r = det - k * (trace * trace)
            if r > threshold:
                if not corner_list:
                    corner_list.append([x, y])
                    output_img[y, x] = (255, 0, 255)
                else:
                    distance = []
                    for corner in corner_list:
                        distance.append(abs(corner[0] - x) + abs(corner[1] - y))
                    if min(distance) > 10:
                        corner_list.append([x, y])
                    output_img[y, x] = (255, 0, 255)

    return corner_list, output_img


video = mpy.VideoFileClip('shapes_video_filtered.mp4')
frame_count = video.reader.nframes
video_fps = video.fps

frame_list = []
for i in range(frame_count):
    frame = video.get_frame(i * 1.0 / video_fps)
    frame_list.append(frame)

frame_list = frame_list[7:-14]


# TODO
# Figure out how to use edge diffs for corner and then shape detection
# Couldn't figure out where to use edge maps but they provide better results when taking differences
# vs the frame differences
diff_list = []
img = np.empty((frame_list[0].shape[0], frame_list[0].shape[1])).astype(np.uint8)
for i in range(len(frame_list)):
    edges = cv2.Canny(frame_list[i], 100, 200)
    diff = cv2.subtract(edges, img)
    img = edges
    diff_list.append(diff)

square_count = 0
pentagon_count = 0
star_count = 0
corners_len = 0
img = np.zeros(frame_list[0].shape).astype(np.uint8)
for i in range(len(frame_list)):
    corners, output = find_harris_corners(cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2RGB), 0.04, 5, 1e9)
    corner_count = len(corners) - corners_len
    corners_len = len(corners)
    if corner_count == 8:
        square_count += 1
    elif corner_count == 9:
        pentagon_count += 1
    elif corner_count == 14:
        star_count += 1
    else:
        print('error', corner_count)
        cv2.imwrite(str(i) + 'error ' + str(corner_count) + ' .png', output)

print(star_count)
print(pentagon_count)
print(square_count)
# Correct counts: square 28 pentagon 22 star 25
