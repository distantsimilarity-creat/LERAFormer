import os
import h5py
import numpy as np
import torch
from torch.utils.data import Sampler


def compute_fg_ratio_cache(train_lines, train_split_dir, cache_path=None, num_classes: int = 2):
    if cache_path and os.path.exists(cache_path):
        try:
            obj = torch.load(cache_path, map_location="cpu")
            if isinstance(obj, dict) and "ratios" in obj:
                ratios = np.asarray(obj["ratios"], dtype=np.float32)
                if ratios.shape[0] == len(train_lines):
                    print(f"[BucketSampler] Loaded fg_ratio cache: {cache_path}")
                    return ratios
        except Exception as e:
            print(f"[BucketSampler] Failed to load cache ({e}), recomputing...")

    ratios = np.zeros((len(train_lines),), dtype=np.float32)

    mask_dir = os.path.join(train_split_dir, "mask")
    for i, line in enumerate(train_lines):
        ann = line.strip().split()[0]
        parts = ann.split(",")
        if len(parts) < 2:
            continue
        mask_name = parts[1].strip()
        mask_path = os.path.join(mask_dir, mask_name)
        try:
            with h5py.File(mask_path, "r") as f:
                key = list(f.keys())[0]
                lab = f[key][:]
            lab = np.asarray(lab)
            lab = np.where(lab >= num_classes, 0, lab)
            ratios[i] = float((lab > 0).mean())
        except Exception as e:
            print(f"[BucketSampler] Warning: fail read {mask_path}: {e}")
            ratios[i] = 0.0

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({"ratios": ratios}, cache_path)
            print(f"[BucketSampler] Saved fg_ratio cache: {cache_path}")
        except Exception as e:
            print(f"[BucketSampler] Failed to save cache: {e}")

    return ratios


def bucketize_ratios(ratios, bins):
    ratios = np.asarray(ratios, dtype=np.float32)
    bins = list(bins) if bins is not None else [0.001, 0.01, 0.05]
    K = len(bins) + 1
    bucket = np.zeros_like(ratios, dtype=np.int64)
    pos = ratios > 0
    bucket[pos] = np.digitize(ratios[pos], bins=bins, right=True) + 1
    bucket = np.clip(bucket, 0, K)
    return bucket


class BucketHardBatchSampler(Sampler):
    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        bucket_id: np.ndarray,
        bucket_weights: list,
        pos_fraction: float = 0.5,
        min_pos_per_batch: int = 2,
        hard_factor: float = 1.0,
        hard_momentum: float = 0.9,
        drop_last: bool = True,
        seed: int = 0,
    ):
        assert num_samples > 0
        assert batch_size > 0
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

        self.bucket_id = np.asarray(bucket_id, dtype=np.int64)
        assert self.bucket_id.shape[0] == self.num_samples

        self.bucket_weights = np.asarray(bucket_weights, dtype=np.float32)
        self.max_bucket = int(self.bucket_id.max())
        if self.bucket_weights.shape[0] <= self.max_bucket:
            raise ValueError(f"bucket_weights length {len(bucket_weights)} must cover bucket_id max {self.max_bucket}")

        self.pos_fraction = float(pos_fraction)
        self.min_pos_per_batch = int(min_pos_per_batch)

        self.hard_factor = float(hard_factor)
        self.hard_momentum = float(hard_momentum)

        self.pos_idx = np.where(self.bucket_id > 0)[0]
        self.neg_idx = np.where(self.bucket_id == 0)[0]

        self.base_w = self.bucket_weights[self.bucket_id]  # shape (N,)
        self.difficulty = np.zeros((self.num_samples,), dtype=np.float32)

        self._g = torch.Generator()
        self._g.manual_seed(int(seed))

        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = int(np.ceil(self.num_samples / float(self.batch_size)))

    def __len__(self):
        return self.num_batches

    def _current_weights(self, idx_array: np.ndarray) -> torch.Tensor:
        base = torch.from_numpy(self.base_w[idx_array]).float()
        if self.hard_factor <= 0:
            return torch.clamp(base, min=1e-8)

        diff = torch.from_numpy(self.difficulty[idx_array]).float()
        mean = diff.mean().clamp(min=1e-6)
        diff_norm = diff / mean
        w = base * (1.0 + self.hard_factor * diff_norm)
        return torch.clamp(w, min=1e-8)

    def __iter__(self):
        n_pos = max(self.min_pos_per_batch, int(round(self.batch_size * self.pos_fraction)))
        n_pos = min(n_pos, self.batch_size)
        n_neg = self.batch_size - n_pos

        pos_pool = self.pos_idx if len(self.pos_idx) > 0 else np.arange(self.num_samples)
        neg_pool = self.neg_idx if len(self.neg_idx) > 0 else np.arange(self.num_samples)

        for _ in range(self.num_batches):
            if n_pos > 0:
                w_pos = self._current_weights(pos_pool)
                pos_sel = torch.multinomial(w_pos, n_pos, replacement=True, generator=self._g).tolist()
                pos_indices = pos_pool[pos_sel]
            else:
                pos_indices = np.empty((0,), dtype=np.int64)

            if n_neg > 0:
                w_neg = self._current_weights(neg_pool)
                neg_sel = torch.multinomial(w_neg, n_neg, replacement=True, generator=self._g).tolist()
                neg_indices = neg_pool[neg_sel]
            else:
                neg_indices = np.empty((0,), dtype=np.int64)

            batch = np.concatenate([pos_indices, neg_indices], axis=0)
            perm = torch.randperm(len(batch), generator=self._g).numpy()
            batch = batch[perm].tolist()
            yield batch

    @torch.no_grad()
    def update_batch(self, indices: torch.Tensor, per_sample_loss: torch.Tensor):
        idx = indices.detach().to("cpu").long().numpy()
        loss = per_sample_loss.detach().to("cpu").float().numpy()
        loss = np.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        loss = np.clip(loss, 0.0, 50.0)

        m = self.hard_momentum
        self.difficulty[idx] = m * self.difficulty[idx] + (1.0 - m) * loss
