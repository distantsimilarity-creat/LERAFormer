import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:
    raise ImportError("Please install h5py: pip install h5py") from e

try:
    import torch
except ImportError as e:
    raise ImportError("Please install torch first.") from e

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DEFAULT_DATASET_ROOT = r"E:\studydata\LERAFormer\data\landslide_dataset"
DEFAULT_TRAIN_LIST = r"E:\studydata\LERAFormer\data\landslide_dataset\TrainData\config\train.txt"
DEFAULT_VAL_LIST   = r"E:\studydata\LERAFormer\data\landslide_dataset\TrainData\config\val.txt"
DEFAULT_TEST_LIST  = r"E:\studydata\LERAFormer\data\landslide_dataset\TrainData\config\test.txt"


def read_pairs_list(list_path: str) -> List[str]:
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"List file not found: {list_path}")

    img_files = []
    with open(list_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split()[0]
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 1:
                img_files.append(parts[0])

    if len(img_files) == 0:
        raise RuntimeError(f"No valid lines in list file: {list_path}")
    return img_files


def load_h5_first_key(h5_path: str) -> np.ndarray:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if len(keys) == 0:
            raise RuntimeError(f"H5 has no keys: {h5_path}")
        arr = f[keys[0]][:]
    return arr


def ensure_hwc(x: np.ndarray, in_channels: int) -> np.ndarray:
    if x.ndim == 2:
        return x[:, :, None]

    if x.ndim != 3:
        raise RuntimeError(f"Unexpected ndim={x.ndim}, shape={x.shape}")

    if x.shape[-1] == in_channels or (x.shape[-1] > 2 and x.shape[0] != in_channels):
        return x

    if x.shape[0] == in_channels and x.shape[-1] != in_channels:
        return np.transpose(x, (1, 2, 0))

    return x


def compute_mean_std_from_split(
    dataset_root: str,
    split_name: str,
    list_path: str,
    in_channels: int,
    max_items: int = 0
) -> Tuple[np.ndarray, np.ndarray, int]:
    split_dir = os.path.join(dataset_root, split_name)
    img_dir = os.path.join(split_dir, "img")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img_dir not found: {img_dir}")

    img_files = read_pairs_list(list_path)
    if max_items and max_items > 0:
        img_files = img_files[: max_items]

    sum_c = np.zeros((in_channels,), dtype=np.float64)
    sumsq_c = np.zeros((in_channels,), dtype=np.float64)
    n_pix = 0

    iterator = img_files
    if tqdm is not None:
        iterator = tqdm(img_files, desc=f"Compute stats [{split_name}]", ncols=100)

    for fn in iterator:
        h5_path = os.path.join(img_dir, fn)
        x = load_h5_first_key(h5_path)
        x = ensure_hwc(x, in_channels)

        if x.shape[-1] < in_channels:
            raise RuntimeError(
                f"{h5_path} has only {x.shape[-1]} channels, but in_channels={in_channels}."
            )

        x = x[:, :, :in_channels].astype(np.float64, copy=False)

        if not np.isfinite(x).all():
            bad = np.logical_not(np.isfinite(x)).sum()
            raise RuntimeError(f"Found NaN/Inf in {h5_path}, bad_count={bad}")

        sum_c += x.sum(axis=(0, 1))
        sumsq_c += (x * x).sum(axis=(0, 1))
        n_pix += x.shape[0] * x.shape[1]

    mean = sum_c / max(n_pix, 1)
    var = sumsq_c / max(n_pix, 1) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32), int(n_pix)


def save_stats(dataset_root: str, name: str, mean: np.ndarray, std: np.ndarray, n_pix: int):
    stats_dir = os.path.join(dataset_root, "norm_stats")
    os.makedirs(stats_dir, exist_ok=True)
    out_path = os.path.join(stats_dir, f"{name}.pt")
    obj = {"mean": mean.tolist(), "std": std.tolist(), "n_pix": int(n_pix)}
    torch.save(obj, out_path)
    print(f"[OK] Saved: {out_path}")
    print(f"     mean({len(mean)}): {mean}")
    print(f"     std ({len(std)}): {std}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Compute per-channel mean/std for H5 dataset (Landslide4Sense-style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT,
                   help="Root dir containing TrainData/ValidData/TestData")
    p.add_argument("--in_channels", type=int, default=14,
                   help="How many channels to use for stats (must match your training input).")
    p.add_argument("--train_list", type=str, default=DEFAULT_TRAIN_LIST)
    p.add_argument("--val_list", type=str, default=DEFAULT_VAL_LIST)
    p.add_argument("--test_list", type=str, default=DEFAULT_TEST_LIST)

    p.add_argument("--train_only", action="store_true",
                   help="Compute TRAIN only and copy to val/test (NOT domain-separated).")

    p.add_argument("--max_items", type=int, default=0,
                   help="Debug: only use first N items (0 = all).")
    return p


def main():
    args = build_parser().parse_args()

    dataset_root = args.dataset_root
    if not os.path.exists(dataset_root):
        print(f"[ERR] dataset_root not found: {dataset_root}")
        sys.exit(1)

    in_channels = int(args.in_channels)

    if args.train_only:
        mean, std, n_pix = compute_mean_std_from_split(
            dataset_root, "TrainData", args.train_list, in_channels, args.max_items
        )
        save_stats(dataset_root, "train", mean, std, n_pix)
        save_stats(dataset_root, "val", mean, std, n_pix)
        save_stats(dataset_root, "test", mean, std, n_pix)
        print("[DONE] train_only mode: copied train stats to val/test.")
    else:
        mean, std, n_pix = compute_mean_std_from_split(
            dataset_root, "TrainData", args.train_list, in_channels, args.max_items
        )
        save_stats(dataset_root, "train", mean, std, n_pix)

        mean, std, n_pix = compute_mean_std_from_split(
            dataset_root, "ValidData", args.val_list, in_channels, args.max_items
        )
        save_stats(dataset_root, "val", mean, std, n_pix)

        mean, std, n_pix = compute_mean_std_from_split(
            dataset_root, "TestData", args.test_list, in_channels, args.max_items
        )
        save_stats(dataset_root, "test", mean, std, n_pix)

        print("[DONE] per-split mode: train/val/test stats computed separately (domain-separated).")

    print("[DONE] norm_stats ready. Your Dataset should auto-load them.")


if __name__ == "__main__":
    main()
