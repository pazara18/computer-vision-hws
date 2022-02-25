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


# Obtain average cat histogram and cdf
red_hists = np.zeros((256, 1))
green_hists = np.zeros((256, 1))
blue_hists = np.zeros((256, 1))

for i in range(180):
    current_path = main_dir + '/cat/cat_' + str(i) + '.png'
    current_image = cv2.imread(current_path)
    red_hist = cv2.calcHist([current_image], [2], np.logical_or(current_image[:, :, 1] < 180, current_image[:, :, 0] > 150).astype(np.uint8), [256], [0, 256])
    green_hist = cv2.calcHist([current_image], [1], np.logical_or(current_image[:, :, 1] < 180, current_image[:, :, 0] > 150).astype(np.uint8), [256], [0, 256])
    blue_hist = cv2.calcHist([current_image], [0], np.logical_or(current_image[:, :, 1] < 180, current_image[:, :, 0] > 150).astype(np.uint8), [256], [0, 256])
    red_hists += red_hist
    green_hists += green_hist
    blue_hists += blue_hist

red_hist = red_hists/180
green_hist = green_hists/180
blue_hist = blue_hists/180

red_cat_avg_cdf = red_hist.cumsum()
red_cat_avg_cdf = red_cat_avg_cdf / red_cat_avg_cdf.max()
green_cat_avg_cdf = green_hist.cumsum()
green_cat_avg_cdf = green_cat_avg_cdf / green_cat_avg_cdf.max()
blue_cat_avg_cdf = blue_hist.cumsum()
blue_cat_avg_cdf = blue_cat_avg_cdf / blue_cat_avg_cdf.max()

# Obtain reference image histogram and cdf
current_path = main_dir + '/histref.png'
current_image = cv2.imread(current_path)
red_ref_hist = cv2.calcHist([current_image], [2], np.logical_or(current_image[:, :, 1] < 180, current_image[:, :, 0] > 150).astype(np.uint8), [256], [0, 256])
green_ref_hist = cv2.calcHist([current_image], [1], np.logical_or(current_image[:, :, 1] < 180, current_image[:, :, 0] > 150).astype(np.uint8), [256], [0, 256])
blue_ref_hist = cv2.calcHist([current_image], [0], np.logical_or(current_image[:, :, 1] < 180, current_image[:, :, 0] > 150).astype(np.uint8), [256], [0, 256])

red_ref_cdf = red_ref_hist.cumsum()
red_ref_cdf = red_ref_cdf / red_ref_cdf.max()
green_ref_cdf = green_ref_hist.cumsum()
green_ref_cdf = green_ref_cdf / green_ref_cdf.max()
blue_ref_cdf = blue_ref_hist.cumsum()
blue_ref_cdf = blue_ref_cdf / blue_ref_cdf.max()


# Takes target image as argument and returns histogram matched target image
def histogram_matching(target):
    LUTr = np.zeros(256)
    gj = 0
    for gi in range(0, 256):
        while red_ref_cdf[gj] < red_cat_avg_cdf[gi] and gj < 256:
            gj += 1
        LUTr[gi] = gj

    LUTg = np.zeros(256)
    gj = 0
    for gi in range(0, 256):
        while green_ref_cdf[gj] < green_cat_avg_cdf[gi] and gj < 256:
            gj += 1
        LUTg[gi] = gj

    LUTb = np.zeros(256)
    gj = 0
    for gi in range(0, 256):
        while blue_ref_cdf[gj] < blue_cat_avg_cdf[gi] and gj < 256:
            gj += 1
        LUTb[gi] = gj

    target[:, :, 0] = LUTb[target[:, :, 0]]
    target[:, :, 1] = LUTg[target[:, :, 1]]
    target[:, :, 2] = LUTr[target[:, :, 2]]

    return target


images_list = []
for i in range(180):
    current_path = main_dir + '/cat/cat_' + str(i) + '.png'
    current_image = cv2.imread(current_path)
    friend_image = current_image.copy()
    friend_image = histogram_matching(friend_image)
    foreground = np.logical_or(current_image[:, :, 1] < 180, current_image[:, :, 0] > 150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = current_image[nonzero_x, nonzero_y, :]
    new_image = background.copy()
    new_image[nonzero_x, nonzero_y, :] = nonzero_cat_values
    nonzero_cat_values = friend_image[nonzero_x, nonzero_y, :]
    new_image[nonzero_x, new_image.shape[1] - nonzero_y - 1, :] = nonzero_cat_values

    new_image = new_image[:, :, [2, 1, 0]]
    images_list.append(new_image)

clip = mpy.ImageSequenceClip(images_list, fps=25)
audio = mpy.AudioFileClip(main_dir + '/selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile('part2_video.mp4', codec='libx264')