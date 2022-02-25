import pickle
import torch
import numpy as np
import moviepy.editor as mpyeditor

with open("stylegan3-t-ffhq-1024x1024.pkl", "rb") as f:
    a = pickle.load(f)

gan = a["G_ema"]

gan.eval()

for param in gan.parameters():
    param.requires_grad = False

first_frame_vector = torch.from_numpy(np.loadtxt('first_frame_vector.txt'))
first_frame_vector = torch.unsqueeze(first_frame_vector, 0)
last_frame_vector = torch.from_numpy(np.loadtxt('last_frame_vector.txt'))
last_frame_vector = torch.unsqueeze(last_frame_vector, 0)

frame_list = []
for i in range(101):
    current_vector = first_frame_vector + (last_frame_vector - first_frame_vector) * i / 100
    current_frame = gan(current_vector, 0).numpy().squeeze()
    current_frame = np.transpose(current_frame, (1, 2, 0))
    current_frame[current_frame > 1] = 1
    current_frame[current_frame < -1] = -1
    current_frame = 255 * (current_frame + 1) / 2
    frame_list.append(current_frame)

clip = mpyeditor.ImageSequenceClip(frame_list, fps=25)
clip.write_videofile('face_video.mp4', codec='libx264')
