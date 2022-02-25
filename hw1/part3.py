import numpy as np
import cv2
import moviepy.editor as moviepy
from matplotlib.path import Path

album_image = cv2.imread('album.png')
cat_img = cv2.imread('cat-headphones.png')
scale = 322 / cat_img.shape[0]
cat_img = cv2.resize(cat_img, (int(cat_img.shape[1] * scale), int(cat_img.shape[0] * scale)))
cat_grayscale = cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY)
# get nonzero coordinates of the cat image
cat_x, cat_y = np.nonzero(cat_grayscale)
cat_x_diff = 161 - (cat_grayscale.shape[1]) // 2
cat_y_diff = 286 - (cat_grayscale.shape[0]) // 2
planes = np.zeros((9, 472, 4, 3))

for i in range(1, 10):
    with open("Plane_" + str(i) + ".txt") as f:
        content = f.readlines()
        for line_id in range(len(content)):
            sel_line = content[line_id]
            sel_line = sel_line.replace(')\n', '').replace("(", '').split(")")

            for point_id in range(4):
                sel_point = sel_line[point_id].split(" ")

                planes[i - 1, line_id, point_id, 0] = float(sel_point[0])
                planes[i - 1, line_id, point_id, 1] = float(sel_point[1])
                planes[i - 1, line_id, point_id, 2] = float(sel_point[2])

images_list = []
for i in range(472):
    depth_LUT = np.full((322, 572), 1000)
    blank_image = np.full((322, 572, 3), 255, np.uint8)
    blank_image[cat_x + cat_x_diff, cat_y + cat_y_diff, :] = cat_img[cat_x, cat_y, :]
    cat_depth = ((max(planes[:, 2].flatten()) + min(planes[:, 2].flatten())) / 2)
    depth_LUT[cat_x + cat_x_diff, cat_y + cat_y_diff] = int(cat_depth)

    for j in range(9):
        # Changed 2 to 3 in [:, 0:2] as depth component is used to determine which plane is in the front
        pts = planes[j, i, :, :].squeeze()[:, 0:3].astype(np.int32)
        temp = np.copy(pts[3, :])
        pts[3, :] = pts[2, :]
        pts[2, :] = temp

        # Loop 4 times as there are 4 correspondences and fill the matrix
        # h is the matrix that maps plane -> album image
        # A * h = 0 and h22 = 1
        # For loop below constructs A matrix by using corner points of the plane and album image
        A = []
        img_width = album_image.shape[1]
        img_height = album_image.shape[0]
        for k in range(4):
            x_s = 0
            y_s = 0

            if k == 1 or k == 2:
                x_s = img_width - 1
            if k == 2 or k == 3:
                y_s = img_height - 1

            x_d = pts[k, 0]
            y_d = pts[k, 1]
            row_1 = [x_d, y_d, 1, 0, 0, 0, -x_s * x_d, -x_s * y_d, -x_s]
            row_2 = [0, 0, 0, x_d, y_d, 1, -y_s * x_d, -y_s * y_d, -y_s]
            A.append(row_1)
            A.append(row_2)

        A.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        M = np.array(A, dtype=np.int32)

        # Solve M * h = b by Least squares method as M might not be invertible
        b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        h = np.linalg.lstsq(M, b, rcond=None)[0]

        # shape h into matrix of form [[h00,h01,h02],[h10,h11,h12],[h20,h21,h22]]
        h = h.reshape(3, 3)

        # Use corner points to construct a polygon and mask coordinates not inside the polygon
        x_min = min([pts[0, 0], pts[1, 0], pts[2, 0], pts[3, 0]])
        x_max = max([pts[0, 0], pts[1, 0], pts[2, 0], pts[3, 0]])
        y_min = min([pts[0, 1], pts[1, 1], pts[2, 1], pts[3, 1]])
        y_max = max([pts[0, 1], pts[1, 1], pts[2, 1], pts[3, 1]])

        plane_coords = np.mgrid[y_min:y_max, x_min:x_max].astype(np.int32)
        plane_coords = np.dstack((plane_coords[1], plane_coords[0]))
        # Corner points to create the polygon
        rectangle_polygon_pts = [[pts[x, 0], pts[x, 1]] for x in range(4)]

        # mask will contain a boolean map of given points in the plane
        rectangle_polygon = Path(rectangle_polygon_pts)
        mask = rectangle_polygon.contains_points(plane_coords.flatten().reshape(-1, 2))

        if len(mask) != 0:
            mask = mask.reshape(-1, x_max - x_min)
            plane_coords = plane_coords[mask]
        else:
            plane_coords = plane_coords.reshape(-1, 2)
        # After operations above plane_coords array contains all points in a particular plane

        # Use depth values from the txt to eliminate plane coords that should not be visible by applying mask
        depth_val = pts[0, 2]
        depth_mask = depth_val < depth_LUT[plane_coords[:, 1], plane_coords[:, 0]]
        plane_coords = plane_coords[depth_mask]
        depth_LUT[plane_coords[:, 1], plane_coords[:, 0]] = depth_val

        # Create ones array and append it to the plane coords for matrix multiplication
        plane_coords = plane_coords.transpose()
        ones_row = np.ones(plane_coords.shape[1], dtype=plane_coords.dtype)
        plane_coords = np.append(plane_coords, [ones_row], axis=0)

        plane_x = plane_coords[0, :]
        plane_y = plane_coords[1, :]

        album_coords = np.matmul(h, plane_coords)

        # Normalize the coordinates by dividing with 3rd axis
        album_x = (album_coords[0, :] / album_coords[2, :]).astype(np.int32)
        album_y = (album_coords[1, :] / album_coords[2, :]).astype(np.int32)

        # Copy transformed album image to the plane in the blank image
        blank_image[plane_y, plane_x, :] = album_image[album_y, album_x, :]

    # Turn RGB to BGR for mpy
    blank_image = blank_image[:, :, [2, 1, 0]]
    images_list.append(blank_image)

clip = moviepy.ImageSequenceClip(images_list, fps=25)
clip.write_videofile("part3_video.mp4", codec="libx264")