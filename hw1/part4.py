import cv2
import numpy as np


def get_clockwise_rotation_matrix(deg):
    return np.array([
        [np.cos(-deg), -np.sin(-deg), 0],
        [np.sin(-deg), np.cos(-deg), 0],
        [0, 0, 1]
    ])


def rotate(image, center_of_rotation, angle):
    new_img = np.full(img.shape, 255, dtype=np.uint8)

    height, width = image.shape[0], image.shape[1]

    T = get_clockwise_rotation_matrix(angle)
    center_of_rotation_x, center_of_rotation_y = center_of_rotation

    x_d, y_d = np.mgrid[0:width, 0:height]
    x_d, y_d = x_d.flatten(), y_d.flatten()
    x_d, y_d = x_d - center_of_rotation_x, y_d - center_of_rotation_y

    p_d = np.stack((x_d, y_d, np.ones(x_d.shape[0])), axis=-1).astype(np.int32).reshape(-1, 3, 1)

    p_s = np.matmul(T, p_d).squeeze().reshape(-1, 3)

    x_s = p_s[:, 0]
    y_s = p_s[:, 1]

    x_s += center_of_rotation_x
    y_s += center_of_rotation_y

    in_img = (x_s < width) & (x_s >= 0) & (y_s < height) & (y_s >= 0)

    x_s = x_s[in_img].astype(np.int32)
    y_s = y_s[in_img].astype(np.int32)

    x_d = p_d[in_img, 0, 0]
    y_d = p_d[in_img, 1, 0]

    x_d += center_of_rotation_x
    y_d += center_of_rotation_y

    new_img[y_d, x_d, :] = img[y_s, x_s, :]

    return new_img


img_file = "album.png"
img = cv2.imread(img_file)

img_1 = rotate(img, (img.shape[1] // 2, img.shape[0] // 2), np.pi / 3)
img_2 = rotate(img, (0, 0), np.pi / 3)

cv2.imwrite("part4_1.png", img_1)
cv2.imwrite("part4_2.png", img_2)

cv2.imshow("original", img)
cv2.imshow("1st rotation", img_1)
cv2.imshow("2nd rotation", img_2)
cv2.waitKey(0)
