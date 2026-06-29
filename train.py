import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, SequentialLR
import matplotlib.pyplot as plt
import h5py

from configs import cfg
from dataloader import build_dataset

import re as _re

def _no_weight_decay(name: str) -> bool:
    lname = name.lower()
    if lname.endswith(".bias"):
        return True
    for k in ["norm", "bn", "layernorm", "batchnorm", "groupnorm", "ln"]:
        if k in lname:
            return True
    return False


def _get_layer_id_swin_like(name: str) -> int:
    lname = name.lower()
    if any(k in lname for k in ["patch_embed", "patchembed", "stem"]):
        return 0
    m = _re.search(r"(?:^|\.)(layers|stages)\.(\d+)", lname)
    if m:
        return 1 + int(m.group(2))
    if any(k in lname for k in ["decoder", "decode", "up", "head", "final", "outc", "seghead", "seg_head", "segmentation_head"]):
        return 10
    return 10


def build_layerwise_param_groups(model: torch.nn.Module,
                                base_lr: float,
                                weight_decay: float,
                                layer_decay: float = 0.75):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(named) == 0:
        return [{"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}]

    layer_ids = [_get_layer_id_swin_like(n) for n, _ in named]
    max_id = int(max(layer_ids))

    groups = {}
    for (n, p), lid in zip(named, layer_ids):
        decay_flag = 0 if _no_weight_decay(n) else 1
        key = (lid, decay_flag)
        if key not in groups:
            scale = layer_decay ** (max_id - lid)
            groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if decay_flag == 0 else weight_decay,
            }
        groups[key]["params"].append(p)

    out = [groups[k] for k in sorted(groups.keys(), key=lambda x: (x[0], x[1]))]
    return out


from dataloader.bucket_hard_sampler import compute_fg_ratio_cache, bucketize_ratios, BucketHardBatchSampler
from model._build_model import build_model
from utils.utils_loop import loop_one_epoch
from utils.ema import ModelEMA


def build_optimizer(cfg, model, optim_name=None):
    base_lr = cfg.optimizer.base_lr
    optim_name = cfg.optimizer.name if optim_name is None else optim_name

    if optim_name == 'adamw':
        use_layer_decay = bool(getattr(cfg.optimizer, 'use_layer_decay', False))
        layer_decay = float(getattr(cfg.optimizer, 'layer_decay', 0.75) or 0.75)

        if use_layer_decay:
            param_groups = build_layerwise_param_groups(
                model,
                base_lr=base_lr,
                weight_decay=cfg.optimizer.weight_decay,
                layer_decay=layer_decay,
            )
            optimizer = optim.AdamW(
                param_groups,
                eps=1e-8,
                betas=(0.9, 0.999),
                lr=base_lr,
                weight_decay=0.0,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                eps=1e-8,
                betas=(0.9, 0.999),
                lr=base_lr,
                weight_decay=cfg.optimizer.weight_decay
            )

    elif optim_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            momentum=0.9,
            nesterov=True,
            lr=base_lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    return optimizer


def build_scheduler(cfg, optimizer, lr_scheduler_name=None):
    scheduler_name = cfg.scheduler.name if lr_scheduler_name is None else lr_scheduler_name

    warmup_epochs = int(getattr(cfg.scheduler, "warmup_epochs", 0) or 0)
    warmup_start_factor = float(getattr(cfg.scheduler, "warmup_start_factor", 0.1) or 0.1)

    if scheduler_name == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.train.total_epoch - warmup_epochs) if warmup_epochs > 0 else cfg.train.total_epoch,
            eta_min=cfg.optimizer.min_lr
        )
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_epochs
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        return main_scheduler

    elif scheduler_name == 'reduce_on_plateau':
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            min_lr=cfg.scheduler.min_lr
        )
        main_scheduler._warmup_epochs = warmup_epochs
        main_scheduler._warmup_start_factor = warmup_start_factor
        return main_scheduler

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def weights_init(model, init_method='normal', init_gain=0.02):
    def init_func(m):
        if isinstance(m, nn.Conv2d):
            if init_method == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_method == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_method == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_method}] is not implemented')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    print(f'initialize network with {init_method} method')
    model.apply(init_func)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def _resolve_dataset_dirs(dataset_path_from_cfg: str):
    p = os.path.normpath(dataset_path_from_cfg)
    base = os.path.basename(p).lower()

    if base in ["traindata", "validdata", "testdata"]:
        dataset_root = os.path.dirname(p)
    else:
        dataset_root = p

    train_path = os.path.join(dataset_root, "TrainData")
    val_path = os.path.join(dataset_root, "ValidData")
    test_path = os.path.join(dataset_root, "TestData")
    return dataset_root, train_path, val_path, test_path


def _build_weighted_sampler(train_lines, train_split_dir, num_classes=2,
                            pos_boost=2.0, ratio_boost=6.0,
                            cache_path=None):
    if cache_path and os.path.exists(cache_path):
        try:
            w = torch.load(cache_path, map_location="cpu")
            if isinstance(w, torch.Tensor) and w.numel() == len(train_lines):
                print(f"[Sampler] Loaded cached weights: {cache_path}")
                return WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        except Exception as e:
            print(f"[Sampler] Failed to load cache: {e}. Recomputing...")

    weights = []
    mask_dir = os.path.join(train_split_dir, "mask")
    for line in train_lines:
        ann = line.strip().split()[0]
        parts = ann.split(",")
        if len(parts) != 2:
            weights.append(1.0)
            continue
        _, mask_name = parts
        mask_path = os.path.join(mask_dir, mask_name)
        try:
            with h5py.File(mask_path, "r") as f:
                m = f[list(f.keys())[0]][:]
            m = np.asarray(m)
            m = m[(m >= 0) & (m < num_classes)]
            if m.size == 0:
                ratio = 0.0
            else:
                ratio = float((m == 1).mean())
        except Exception:
            ratio = 0.0

        has_fg = 1.0 if ratio > 0 else 0.0
        w = 1.0 + pos_boost * has_fg + ratio_boost * ratio
        weights.append(float(w))

    w_tensor = torch.tensor(weights, dtype=torch.double)
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(w_tensor, cache_path)
            print(f"[Sampler] Saved weights cache to: {cache_path}")
        except Exception as e:
            print(f"[Sampler] Failed to save cache: {e}")

    return WeightedRandomSampler(w_tensor, num_samples=len(w_tensor), replacement=True)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seed(0, deterministic=True)
    model_type = cfg.model.model_type
    num_classes = cfg.model.num_classes

    w_pos = getattr(getattr(cfg, "loss", {}), "w_pos", 3.0) if hasattr(cfg, "loss") else 3.0
    cls_weights = torch.tensor([1.0, float(w_pos)], dtype=torch.float32)
    if cfg.train.cuda:
        cls_weights = cls_weights.cuda()

    dataset_path_cfg = cfg.dataset.dataset_path
    dataset_root, train_path, val_path, _ = _resolve_dataset_dirs(dataset_path_cfg)

    train_lines_path = cfg.dataset.train_lines
    val_lines_path = cfg.dataset.val_lines

    isAug = cfg.dataloader.isOnLineAug
    batch_size = cfg.dataloader.batch_size
    num_workers = cfg.dataloader.num_workers
    input_shape = cfg.dataloader.input_shape
    in_channels = cfg.dataloader.in_channels
    keep_idx = getattr(cfg.dataloader, "channel_keep_idx", None)
    derived_indices = getattr(cfg.dataloader, "derived_indices", None)
    derived_replace_channels = getattr(cfg.dataloader, "derived_replace_channels", None)
    derived_stats = getattr(cfg.dataloader, "derived_stats", None)
    cuda = cfg.train.cuda

    end_epoch = cfg.train.total_epoch
    resume_path = cfg.train.ckpt_resume
    ckpt_savpath = cfg.train.ckpt_savepath
    freeze_param = cfg.train.freeze_param

    time_str = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    save_path = os.path.join(ckpt_savpath, "loss_" + str(time_str))
    os.makedirs(save_path, exist_ok=True)

    with open(train_lines_path, "r", encoding="utf-8", errors="ignore") as f:
        train_lines = f.readlines()
        random.shuffle(train_lines)

    with open(val_lines_path, "r", encoding="utf-8", errors="ignore") as f:
        val_lines = f.readlines()

    train_data = build_dataset(
        train_lines, input_shape, in_channels, num_classes, isAug, train_path,
        channel_keep_idx=keep_idx,
        derived_indices=derived_indices,
        derived_replace_channels=derived_replace_channels,
        derived_stats=derived_stats,
    )
    val_data = build_dataset(
        val_lines, input_shape, in_channels, num_classes, False, val_path,
        channel_keep_idx=keep_idx,
        derived_indices=derived_indices,
        derived_replace_channels=derived_replace_channels,
        derived_stats=derived_stats,
    )

    sampler_cfg = getattr(cfg, "sampler", None)
    sampler_enable = bool(getattr(sampler_cfg, "enable", False)) if sampler_cfg else False
    sampler_strategy = str(getattr(sampler_cfg, "strategy", "weighted")).lower() if sampler_cfg else "weighted"

    sampler_cache = getattr(sampler_cfg, "cache_path", None) if sampler_cfg else None
    if sampler_cache is None:
        sampler_cache = os.path.join(os.path.dirname(train_lines_path), "train_fg_ratio_cache.pt")

    train_sampler = None
    train_batch_sampler = None

    if sampler_enable and sampler_strategy == "bucket_hard":
        train_data = build_dataset(
            train_lines, input_shape, in_channels, num_classes, isAug, train_path,
            return_index=True,
            channel_keep_idx=keep_idx,
            derived_indices=derived_indices,
            derived_replace_channels=derived_replace_channels,
            derived_stats=derived_stats,
        )

        bucket_bins = getattr(sampler_cfg, "bucket_bins", [0.001, 0.01, 0.05])
        default_bucket_weights = [0.2, 4.0, 6.0, 4.0, 2.0]
        bucket_weights = list(getattr(sampler_cfg, "bucket_weights", default_bucket_weights))

        ratios = compute_fg_ratio_cache(train_lines, train_path, cache_path=sampler_cache, num_classes=num_classes)
        bucket_id = bucketize_ratios(ratios, bucket_bins)

        K = int(bucket_id.max())
        if len(bucket_weights) <= K:
            bucket_weights = bucket_weights + [float(bucket_weights[-1])] * (K + 1 - len(bucket_weights))

        pos_fraction = float(getattr(sampler_cfg, "pos_fraction", 0.5))
        min_pos_per_batch = int(getattr(sampler_cfg, "min_pos_per_batch", 2))
        hard_factor = float(getattr(sampler_cfg, "hard_factor", 1.0))
        hard_momentum = float(getattr(sampler_cfg, "hard_momentum", 0.9))

        train_batch_sampler = BucketHardBatchSampler(
            num_samples=len(train_lines),
            batch_size=batch_size,
            bucket_id=bucket_id,
            bucket_weights=bucket_weights,
            pos_fraction=pos_fraction,
            min_pos_per_batch=min_pos_per_batch,
            hard_factor=hard_factor,
            hard_momentum=hard_momentum,
            drop_last=True,
            seed=0,
        )

    elif sampler_enable:
        sampler_pos_boost = float(getattr(sampler_cfg, "pos_boost", 2.0))
        sampler_ratio_boost = float(getattr(sampler_cfg, "ratio_boost", 6.0))
        weights_cache = os.path.join(os.path.dirname(train_lines_path), "train_sampler_weights.pt")
        train_sampler = _build_weighted_sampler(
            train_lines, train_path, num_classes=num_classes,
            pos_boost=sampler_pos_boost, ratio_boost=sampler_ratio_boost,
            cache_path=weights_cache
        )

    persistent = True if num_workers and num_workers > 0 else False

    train_loader = DataLoader(
        train_data,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent
    ) if train_batch_sampler is not None else DataLoader(
        train_data,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent
    )

    model = build_model(model_type)
    if cuda:
        model.to(device)

    optimizer = build_optimizer(cfg, model, None)
    lr_scheduler = build_scheduler(cfg, optimizer, None)

    base_lr = float(cfg.optimizer.base_lr)
    warmup_epochs = 0
    warmup_start_factor = 0.1
    if isinstance(lr_scheduler, ReduceLROnPlateau):
        warmup_epochs = int(getattr(lr_scheduler, "_warmup_epochs", 0) or 0)
        warmup_start_factor = float(getattr(lr_scheduler, "_warmup_start_factor", 0.1) or 0.1)

    ema_enable = False
    ema_decay = 0.9999
    ema_eval = True
    ema_save = True
    if hasattr(cfg, "ema"):
        ema_enable = bool(getattr(cfg.ema, "enable", False))
        ema_decay = float(getattr(cfg.ema, "decay", ema_decay))
        ema_eval = bool(getattr(cfg.ema, "eval", ema_eval))
        ema_save = bool(getattr(cfg.ema, "save", ema_save))

    ema = ModelEMA(model, decay=ema_decay) if ema_enable else None
    if ema_enable:
        print(f"[EMA] enabled: decay={ema_decay}, eval={ema_eval}, save={ema_save}")
    start_epoch = 0
    if cfg.train.resume:
        checkpoint = torch.load(resume_path, map_location=device)
        start_epoch = checkpoint.get('epoch', 0)
        model.load_state_dict(checkpoint['model'])
        if checkpoint.get('optimizer', None) is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('lr_scheduler', None) is not None and hasattr(lr_scheduler, "load_state_dict"):
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if ema is not None and checkpoint.get("ema", None) is not None:
            try:
                ema.load_state_dict(checkpoint["ema"])
                print("[EMA] resumed from checkpoint.")
            except Exception as e:
                print(f"[EMA] resume failed: {e}")

    if freeze_param:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.freeze_param()

    val_thr_list = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    step_train = len(train_loader)
    step_val = len(val_loader)

    for epoch in range(start_epoch, end_epoch):
        print('Start Epoch')
        train_loss, val_loss, val_iou, val_f1 = loop_one_epoch(
            model, save_path, optimizer, lr_scheduler, epoch,
            end_epoch, step_train, step_val, train_loader, val_loader,
            cuda, cls_weights, num_classes,
            use_amp=True,
            grad_clip_norm=1.0,
            stop_on_nonfinite=True,
            val_thr_list=val_thr_list,
            save_best_thr=True,
            select_best_thr_by="f1",
            ema=ema,
            ema_eval=ema_eval,
            ema_save=ema_save
        )
        print('Finish Epoch')

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warm = (epoch + 1) / float(warmup_epochs)
                factor = warmup_start_factor + (1.0 - warmup_start_factor) * warm
                new_lr = base_lr * factor
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
            else:
                lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
    epochs, train_losses, val_losses, val_ious, val_f1s = [], [], [], [], []
    log_file = os.path.join(save_path, 'log.txt')
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(',')
                epoch = int(parts[0].split()[1])
                train_loss = float(parts[1].split(':')[1])
                val_loss = float(parts[2].split(':')[1])
                val_iou = float(parts[3].split(':')[1])
                val_f1 = float(parts[4].split(':')[1])
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_ious.append(val_iou)
                val_f1s.append(val_f1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_ious, label='Val IoU')
        plt.plot(epochs, val_f1s, label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation IoU and F1 Score')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_metrics.png'))
        plt.show()


if __name__ == '__main__':
    main()