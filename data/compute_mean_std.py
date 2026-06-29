import h5py
import torch
import numpy as np
import os

img_dir = ''
images = []
for img_name in os.listdir(img_dir):
    if img_name.startswith('image_') and img_name.endswith('.h5'):
        with h5py.File(os.path.join(img_dir, img_name), 'r') as f:
            img = f[list(f.keys())[0]][:]
            images.append(img)
images = np.stack(images, axis=0)
images = torch.tensor(images, dtype=torch.float32)
means = images.mean(dim=(0, 1, 2)).tolist()
stds = images.std(dim=(0, 1, 2)).tolist()
print("Means:", means)
print("Stds:", stds)