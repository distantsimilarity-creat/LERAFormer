import os
import re

train_img_dir = 'E:/studydata/LERAFormer/data/landslide_dataset/TrainData/img/'
train_mask_dir = 'E:/studydata/LERAFormer/data/landslide_dataset/TrainData/mask/'
valid_img_dir = 'E:/studydata/LERAFormer/data/landslide_dataset/ValidData/img/'
valid_mask_dir = 'E:/studydata/LERAFormer/data/landslide_dataset/ValidData/mask/'
test_img_dir = 'E:/studydata/LERAFormer/data/landslide_dataset/TestData/img/'
test_mask_dir = 'E:/studydata/LERAFormer/data/landslide_dataset/TestData/mask/'
config_dir = 'E:/studydata/LERAFormer/data/landslide_dataset/TrainData/config/'

def numerical_sort(files):
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
    return sorted(files, key=extract_number)

train_img_files = [f for f in os.listdir(train_img_dir) if f.startswith('image_') and f.endswith('.h5')]
train_mask_files = [f for f in os.listdir(train_mask_dir) if f.startswith('mask_') and f.endswith('.h5')]
valid_img_files = [f for f in os.listdir(valid_img_dir) if f.startswith('image_') and f.endswith('.h5')]
valid_mask_files = [f for f in os.listdir(valid_mask_dir) if f.startswith('mask_') and f.endswith('.h5')]
test_img_files = [f for f in os.listdir(test_img_dir) if f.startswith('image_') and f.endswith('.h5')]
test_mask_files = [f for f in os.listdir(test_mask_dir) if f.startswith('mask_') and f.endswith('.h5')]

train_img_files = numerical_sort(train_img_files)
train_mask_files = numerical_sort(train_mask_files)
valid_img_files = numerical_sort(valid_img_files)
valid_mask_files = numerical_sort(valid_mask_files)
test_img_files = numerical_sort(test_img_files)
test_mask_files = numerical_sort(test_mask_files)

assert len(train_img_files) == len(train_mask_files), f"训练集图像和标签数量不匹配: {len(train_img_files)} vs {len(train_mask_files)}"
assert len(valid_img_files) == len(valid_mask_files), f"验证集图像和标签数量不匹配: {len(valid_img_files)} vs {len(valid_mask_files)}"
assert len(test_img_files) == len(test_mask_files), f"测试集图像和标签数量不匹配: {len(test_img_files)} vs {len(test_mask_files)}"
assert len(train_img_files) == 3799, f"训练集图像数量不正确: 预期 3799，实际 {len(train_img_files)}"

train_files = list(zip(train_img_files, train_mask_files))
valid_files = list(zip(valid_img_files, valid_mask_files))
test_files = list(zip(test_img_files, test_mask_files))

os.makedirs(config_dir, exist_ok=True)

with open(os.path.join(config_dir, 'train.txt'), 'w') as f:
    for img, mask in train_files:
        f.write(f"{img},{mask}\n")

with open(os.path.join(config_dir, 'val.txt'), 'w') as f:
    for img, mask in valid_files:
        f.write(f"{img},{mask}\n")

with open(os.path.join(config_dir, 'test.txt'), 'w') as f:
    for img, mask in test_files:
        f.write(f"{img},{mask}\n")

print(f"Generated: train.txt ({len(train_files)} pairs), val.txt ({len(valid_files)} pairs), test.txt ({len(test_files)} pairs)")