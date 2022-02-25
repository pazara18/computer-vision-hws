import numpy as np
import os
import cv2
import moviepy.editor as mpy

main_dir = os.getcwd()
background = cv2.imread('Malibu.jpg')

background_height = background.shape[0]
background_width = background.shape[1]
ratio = 360/background_height

background = cv2.resize(background, (int(background_width*ratio), 360))

images_list = []
for i in range(180):
    current_path = main_dir + '/cat/cat_' + str(i) + '.png'
    current_image = cv2.imread(current_path)

    foreground = np.logical_or(current_image[:, :, 1] < 180, current_image[:, :, 0] > 150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = current_image[nonzero_x, nonzero_y, :]
    new_image = background.copy()
    new_image[nonzero_x, nonzero_y, :] = nonzero_cat_values
    new_image[nonzero_x, new_image.shape[1] - nonzero_y - 1, :] = nonzero_cat_values
    new_image = new_image[:, :, [2, 1, 0]]

    images_list.append(new_image)

clip = mpy.ImageSequenceClip(images_list, fps=25)
audio = mpy.AudioFileClip(main_dir + '/selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile('part1_video.mp4', codec='libx264')