import pickle
import torch
import numpy as np
import cv2

with open("stylegan3-t-ffhq-1024x1024.pkl", "rb") as f:
    a = pickle.load(f)

gan = a["G_ema"]

gan.eval()

for param in gan.parameters():
    param.requires_grad = False

z1 = torch.randn(1, 512)
z2 = torch.randn(1, 512)
np.savetxt('first_frame_vector.txt', z1)
np.savetxt('last_frame_vector.txt', z2)
i = 1
for z in [z1, z2]:
    img = gan(z, 0).numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))
    img[img > 1] = 1
    img[img < -1] = -1
    img = 255 * (img + 1) / 2
    cv2.imwrite('test' + str(i) + '.png', img[:, :, [2, 1, 0]])
    i += 1
