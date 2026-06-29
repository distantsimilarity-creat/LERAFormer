import h5py
import os
import numpy as np

mask_dir = 'E:/studydata/LERAFormer/data/landslide_dataset/TrainData/mask/'
pos_count, total_count = 0, 0

for mask_name in os.listdir(mask_dir):
    if mask_name.startswith('mask_') and mask_name.endswith('.h5'):
        with h5py.File(os.path.join(mask_dir, mask_name), 'r') as f:
            mask = f[list(f.keys())[0]][:]
            pos_count += np.sum(mask == 1)
            total_count += mask.size

pos_ratio = pos_count / total_count
neg_ratio = 1 - pos_ratio
print(f"Positive ratio: {pos_ratio:.4f}, Negative ratio: {neg_ratio:.4f}")

cls_weights = torch.tensor([neg_ratio, pos_ratio], dtype=torch.float32)
print("Class weights:", cls_weights)
