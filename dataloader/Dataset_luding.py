import os
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class GeneralDataset(Dataset):
    def __init__(
        self,
        annotation_lines,
        in_shape,
        in_channels,
        num_classes,
        random,
        dataset_path,
        return_index: bool = False,
        channel_keep_idx=None,
        derived_indices=None,
        derived_replace_channels=None,
        derived_stats=None,
    ):
        self.annotation_lines = annotation_lines
        self.path = dataset_path
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.input_shape = in_shape
        self.random = bool(random)
        self.return_index = bool(return_index)
        self.channel_keep_idx = channel_keep_idx
        if self.channel_keep_idx is not None:
            m = torch.zeros(int(in_channels), dtype=torch.float32)
            m[self.channel_keep_idx] = 1.0
            self.channel_mask_chw = m.view(int(in_channels), 1, 1)
        else:
            self.channel_mask_chw = None
        self.derived_indices = list(derived_indices) if derived_indices is not None else None
        self.derived_replace_channels = list(derived_replace_channels) if derived_replace_channels is not None else None
        self.derived_stats = dict(derived_stats) if isinstance(derived_stats, dict) else None
        self.terrain_channels = 2 if self.in_channels >= 2 else 0

        self.means = torch.tensor(
            [0.9257, 0.9227, 0.9541, 0.9596, 1.0228, 1.0426, 1.0358, 1.0468, 1.1699,
             1.1736, 1.0495, 1.0370, 1.2511, 1.6495],
            dtype=torch.float32
        )[: self.in_channels]
        self.stds = torch.tensor(
            [0.1410, 0.2207, 0.3184, 0.5724, 0.4601, 0.4465, 0.4651, 0.4948, 0.5133,
             0.6836, 0.5323, 0.6628, 0.6784, 1.0727],
            dtype=torch.float32
        )[: self.in_channels]
        base = os.path.basename(os.path.normpath(self.path)).lower()
        self.split = base if base in ["traindata", "validdata", "testdata"] else None
        self.dataset_root = os.path.dirname(self.path) if self.split else os.path.dirname(self.path)
        self._try_load_split_norm_stats()

    def _try_load_split_norm_stats(self):
        try:
            if not self.split:
                return
            stats_dir = os.path.join(self.dataset_root, "norm_stats")
            if not os.path.isdir(stats_dir):
                return

            name = {"traindata": "train", "validdata": "val", "testdata": "test"}[self.split]
            stats_path = os.path.join(stats_dir, f"{name}.pt")
            if not os.path.isfile(stats_path):
                return

            stats = torch.load(stats_path, map_location="cpu")
            mean = torch.as_tensor(stats["mean"], dtype=torch.float32)[: self.in_channels]
            std = torch.as_tensor(stats["std"], dtype=torch.float32)[: self.in_channels]
            std = torch.clamp(std, min=1e-6)
            self.means = mean
            self.stds = std
            print(f"[NormStats] Loaded {stats_path}")
        except Exception as e:
            print(f"[NormStats] Failed to load split stats: {e}. Using fallback mean/std.")

    def _compute_index(self, name: str, img_raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        n = name.upper()
        B2 = img_raw[..., 1]
        B4 = img_raw[..., 3]
        B8 = img_raw[..., 7]
        B11 = img_raw[..., 10]
        B12 = img_raw[..., 11]

        if n == "NDVI":
            out = (B8 - B4) / (B8 + B4 + eps)
        elif n == "NDMI":
            out = (B8 - B11) / (B8 + B11 + eps)
        elif n == "NBR":
            out = (B8 - B12) / (B8 + B12 + eps)
        elif n == "BSI":
            out = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + eps)
        else:
            raise ValueError(f"Unknown derived index: {name}")
        return out.clamp(-1.0, 1.0).to(torch.float32)

    def _standardize_index(self, name: str, idx: torch.Tensor) -> torch.Tensor:
        if not self.derived_stats:
            return idx
        key = name.upper()
        if key not in self.derived_stats:
            return idx
        m = float(self.derived_stats[key].get("mean", 0.0))
        s = float(self.derived_stats[key].get("std", 1.0))
        s = max(s, 1e-6)
        return (idx - m) / s

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        img, label = self.h5_ts(index)
        if self.random:
            img, label = self.ts_augmentation(img, label)
        tensor_img, tensor_label, tensor_oh_labels = self.process(img, label)
        if self.return_index:
            return tensor_img, tensor_label, tensor_oh_labels, index
        return tensor_img, tensor_label, tensor_oh_labels

    def _resolve_paths(self, img_name: str, label_name: str):
        if self.split in ["traindata", "validdata", "testdata"]:
            img_path = os.path.join(self.path, "img", img_name)
            label_path = os.path.join(self.path, "mask", label_name)
            return img_path, label_path

        base_path = os.path.dirname(self.path)
        train_img_dir = os.path.join(self.path, "img")
        if os.path.isdir(train_img_dir) and (img_name in os.listdir(train_img_dir)):
            img_path = os.path.join(self.path, "img", img_name)
            label_path = os.path.join(self.path, "mask", label_name)
            return img_path, label_path

        val_img_dir = os.path.join(base_path, "ValidData", "img")
        if os.path.isdir(val_img_dir) and (img_name in os.listdir(val_img_dir)):
            img_path = os.path.join(base_path, "ValidData", "img", img_name)
            label_path = os.path.join(base_path, "ValidData", "mask", label_name)
            return img_path, label_path

        img_path = os.path.join(base_path, "TestData", "img", img_name)
        label_path = os.path.join(base_path, "TestData", "mask", label_name)
        return img_path, label_path

    def h5_ts(self, index):
        annotation_line = self.annotation_lines[index].split()[0]
        img_name = annotation_line.split(',')[0]
        label_name = annotation_line.split(',')[1]

        img_path, label_path = self._resolve_paths(img_name, label_name)

        with h5py.File(img_path, 'r') as img_h5:
            img = img_h5[list(img_h5.keys())[0]][:]  # (H, W, C)
        with h5py.File(label_path, 'r') as label_h5:
            label = label_h5[list(label_h5.keys())[0]][:]  # (H, W)

        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        label[label >= self.num_classes] = 0

        means = self.means.to(img.device).view(1, 1, -1)
        stds = self.stds.to(img.device).view(1, 1, -1)
        img_z = (img[..., : self.in_channels] - means) / (stds + 1e-6)

        if self.derived_indices is not None:
            replace_ch = self.derived_replace_channels
            if replace_ch is None:
                raise ValueError("derived_replace_channels must be provided when derived_indices is set")
            if len(replace_ch) != len(self.derived_indices):
                raise ValueError("derived_replace_channels and derived_indices must have the same length")

            for name, ch in zip(self.derived_indices, replace_ch):
                if int(ch) >= self.in_channels:
                    raise ValueError(f"Replace channel {ch} is out of range for in_channels={self.in_channels}")
                idx = self._compute_index(name, img[..., : self.in_channels])
                idx = self._standardize_index(name, idx)
                img_z[..., int(ch)] = idx

        img = img_z

        return img, label

    def random_crop(self, img, lbl, h, w):
        i = torch.randint(0, img.shape[0] - h + 1, (1,)).item()
        j = torch.randint(0, img.shape[1] - w + 1, (1,)).item()
        return img[i:i + h, j:j + w, :], lbl[i:i + h, j:j + w]

    def random_erasing(self, img, p=0.5, sl=0.02, sh=0.2, r1=0.3):
        if torch.rand(1) > p:
            return img
        H, W = img.shape[:2]
        area = H * W
        for _ in range(100):
            target_area = torch.FloatTensor(1).uniform_(sl, sh).item() * area
            aspect_ratio = torch.FloatTensor(1).uniform_(r1, 1 / r1).item()
            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))
            if w < W and h < H:
                i = torch.randint(0, H - h + 1, (1,)).item()
                j = torch.randint(0, W - w + 1, (1,)).item()
                img[i:i + h, j:j + w, :] = torch.randn(h, w, img.shape[2], device=img.device, dtype=img.dtype)
                return img
        return img

    def spectral_perturb(self, img, p=0.7):
        if torch.rand(1).item() > p:
            return img

        c = img.shape[-1]
        ms_end = max(0, c - self.terrain_channels)
        if ms_end <= 0:
            return img
        gain = 1.0 + torch.empty((ms_end,), device=img.device).uniform_(-0.10, 0.10)
        bias = torch.empty((ms_end,), device=img.device).uniform_(-0.05, 0.05)

        sigma = float(torch.empty(1).uniform_(0.0, 0.03).item())

        out = img.clone()
        out[..., :ms_end] = out[..., :ms_end] * gain.view(1, 1, -1) + bias.view(1, 1, -1)
        if sigma > 0:
            out[..., :ms_end] = out[..., :ms_end] + sigma * torch.randn_like(out[..., :ms_end])
        return out

    def ts_augmentation(self, image, label):
        if torch.rand(1) > 0.5:
            image = torch.flip(image, [0])
            label = torch.flip(label, [0])
        if torch.rand(1) > 0.5:
            image = torch.flip(image, [1])
            label = torch.flip(label, [1])
        if torch.rand(1) > 0.5:
            image = torch.rot90(image, 1, [0, 1])
            label = torch.rot90(label, 1, [0, 1])
        image = self.spectral_perturb(image, p=0.7)

        if torch.rand(1) > 0.5:
            image, label = self.random_crop(image, label, self.input_shape[0], self.input_shape[1])
        if torch.rand(1) > 0.5:
            image = self.random_erasing(image)
        return image, label

    def process(self, in_img, label):
        h, w = self.input_shape
        dh = in_img.shape[0]
        img = in_img[:, :, : self.in_channels]
        img = img.permute(2, 0, 1)

        if self.channel_mask_chw is not None:
            img = img * self.channel_mask_chw

        if h != dh:
            img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
            label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze(0).squeeze(0)

        tensor_img = img
        tensor_label = label.long()
        tensor_oh_labels = F.one_hot(tensor_label, self.num_classes + 1)
        return tensor_img, tensor_label, tensor_oh_labels