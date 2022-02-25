import numpy as np
import cv2
import moviepy.video.io.VideoFileClip as mpy
import moviepy.editor as mpyeditor


def median_filter(src, window_size):
    padding = window_size // 2
    offset_x, offset_y = np.mgrid[-padding:padding+1, -padding:padding+1]
    offset_x, offset_y = offset_x.ravel(), offset_y.ravel()
    padded_img = cv2.copyMakeBorder(src, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    h, w, _ = padded_img.shape
    x = np.arange(padding, w - padding)
    y = np.arange(padding, h - padding)
    xv, yv = np.meshgrid(x, y)
    xv = np.repeat(xv[..., np.newaxis], window_size * window_size, axis=2)
    yv = np.repeat(yv[..., np.newaxis], window_size * window_size, axis=2)
    xv += offset_x
    yv += offset_y
    kernel = padded_img[yv, xv]
    median_frame = np.median(kernel, axis=2).astype(np.uint8)
    return median_frame


video = mpy.VideoFileClip('shapes_video.mp4')
frame_count = video.reader.nframes
video_fps = video.fps
frame_list = []
for i in range(frame_count):
    frame = video.get_frame(i * 1.0 / video_fps)
    filtered = median_filter(frame, 3)
    frame_list.append(filtered)

clip = mpyeditor.ImageSequenceClip(frame_list, fps=25)
clip.write_videofile('shapes_video_filtered.mp4', codec='libx264')
