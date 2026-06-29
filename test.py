import os
import time
import argparse

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from utils.utils_loss import build_loss, gen_matrix, matrix2index
from model._build_model import build_model
from dataloader import build_dataset
from configs import cfg


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
    return train_path, val_path, test_path


def _load_thr_near_checkpoint(model_path: str, default: float = 0.5):
    thr_file = os.path.join(os.path.dirname(model_path), "best_thr.txt")
    if os.path.exists(thr_file):
        try:
            thr = float(open(thr_file, "r", encoding="utf-8").read().strip())
            return thr, thr_file
        except Exception:
            return default, thr_file
    return default, thr_file


def _maybe_disable_jit_fusers():
    try:
        torch.jit.set_fusion_strategy([("STATIC", 0)])
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", type=float, default=None,
                        help="前景阈值（覆盖 best_thr.txt）。例如 --thr 0.70")
    parser.add_argument("--sweep_thr", action="store_true",
                        help="在测试集上扫阈值（仅用于分析/报告 best-thr），默认关闭。")
    parser.add_argument("--thr_min", type=float, default=0.35)
    parser.add_argument("--thr_max", type=float, default=0.85)
    parser.add_argument("--thr_step", type=float, default=0.01)

    args = parser.parse_args()

    model_type = cfg.model.model_type
    num_classes = cfg.model.num_classes

    dataset_path_cfg = cfg.dataset.dataset_path
    _, _, test_path = _resolve_dataset_dirs(dataset_path_cfg)

    test_lines_path = cfg.dataset.test_lines
    input_shape = cfg.dataloader.input_shape
    in_channels = cfg.dataloader.in_channels
    batch_size = cfg.dataloader.batch_size
    keep_idx = getattr(cfg.dataloader, "channel_keep_idx", None)
    derived_indices = getattr(cfg.dataloader, "derived_indices", None)
    derived_replace_channels = getattr(cfg.dataloader, "derived_replace_channels", None)
    derived_stats = getattr(cfg.dataloader, "derived_stats", None)
    cuda = bool(cfg.train.cuda) and torch.cuda.is_available()

    with open(test_lines_path, "r", encoding="utf-8", errors="ignore") as f:
        test_lines = f.readlines()

    test_dataset = build_dataset(
        test_lines,
        input_shape,
        in_channels,
        num_classes,
        False,
        test_path,
        channel_keep_idx=keep_idx,
        derived_indices=derived_indices,
        derived_replace_channels=derived_replace_channels,
        derived_stats=derived_stats,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=1,
        drop_last=False
    )

    device = torch.device("cuda:0" if cuda else "cpu")

    model = build_model(model_type).to(device).eval()

    ckpt = cfg.train.ckpt_test
    model_path = ckpt if os.path.isabs(ckpt) else os.path.join(cfg.train.ckpt_savepath, ckpt)

    checkpoint = torch.load(model_path, map_location=device)

    use_ema_for_test = True
    if hasattr(cfg, "ema") and hasattr(cfg.ema, "use_for_test"):
        use_ema_for_test = bool(cfg.ema.use_for_test)

    if use_ema_for_test and (isinstance(checkpoint, dict) and checkpoint.get("ema", None) is not None):
        base_sd = checkpoint["model"]
        ema_sd = checkpoint["ema"]
        merged_sd = dict(base_sd)
        merged_sd.update(ema_sd)
        model.load_state_dict(merged_sd, strict=True)
        print("[EMA] Using EMA params + base BN buffers.")
    else:
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

    print(f"{model_path} model, and classes loaded.")

    if args.thr is not None:
        thr = float(args.thr)
        print(f"[THR] use override threshold = {thr:.4f} (from CLI)")
    else:
        thr, thr_file = _load_thr_near_checkpoint(model_path, default=0.5)
        if os.path.exists(thr_file):
            print(f"[THR] use threshold = {thr:.4f} (from {thr_file})")
        else:
            print(f"[THR] best_thr.txt not found, fallback threshold = {thr:.4f}")

    _maybe_disable_jit_fusers()

    example_input = torch.randn(1, in_channels, *input_shape, device=device)
    try:
        model = torch.jit.trace(model, example_input)
        model = model.eval()
        print("Model JIT compiled for faster inference.")
    except Exception as e:
        print(f"JIT tracing failed: {e}. Proceeding without JIT.")

    epoch_step_test = len(test_lines) // batch_size + (1 if len(test_lines) % batch_size != 0 else 0)

    if args.sweep_thr:
        thr_list = []
        t0 = float(args.thr_min)
        tmax = float(args.thr_max)
        step = float(args.thr_step)
        while t0 <= tmax + 1e-9:
            thr_list.append(round(t0, 6))
            t0 += step
        cf_mtx_dict = {t: torch.zeros((num_classes, num_classes), device=device) for t in thr_list}
        print(f"[THR] sweep enabled: {len(thr_list)} thresholds from {thr_list[0]:.2f} to {thr_list[-1]:.2f} step={step:.3f}")
    else:
        cf_mtx = torch.zeros((num_classes, num_classes), device=device)

    if cuda:
        dummy_input = torch.randn(batch_size, in_channels, *input_shape, device=device)
        with torch.no_grad():
            for _ in range(5):
                with autocast():
                    _ = model(dummy_input)
        torch.cuda.synchronize()

    test_loss = 0.0
    t1 = time.time()

    with tqdm(total=epoch_step_test, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(test_loader):
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                imgs, labels, onehot_labels, _ = batch
            else:
                imgs, labels, onehot_labels = batch

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            onehot_labels = onehot_labels.to(device, non_blocking=True)

            with torch.no_grad():
                with autocast():
                    outputs = model(imgs).float()

                loss = build_loss(outputs, labels, onehot_labels, loss_cfg=getattr(cfg, "loss", None))
                test_loss += float(loss.item())

                if args.sweep_thr:
                    for t_thr in thr_list:
                        mtx = gen_matrix(outputs, labels, num_classes=num_classes, thr=float(t_thr))[0]
                        cf_mtx_dict[t_thr] += mtx
                else:
                    mtx = gen_matrix(outputs, labels, num_classes=num_classes, thr=float(thr))[0]
                    cf_mtx += mtx

            pbar.update(1)

    if cuda:
        torch.cuda.synchronize()
    t2 = time.time()
    FPS = (iteration + 1) / (t2 - t1)
    print("FPS:       ", FPS)

    if args.sweep_thr:
        best = None
        for t_thr in thr_list:
            m_acc, m_prec, m_rec, m_f1, m_iou, m_miou, m_mcc = matrix2index(cf_mtx_dict[t_thr])
            key = m_f1
            if best is None or key > best[0]:
                best = (key, t_thr, (m_acc, m_prec, m_rec, m_f1, m_iou, m_miou, m_mcc))
        thr_best = float(best[1])
        test_acc, test_precision, test_recall, test_f_score, test_iou, test_miou, test_mcc = best[2]
        print(f"[THR] sweep best thr = {thr_best:.6f} (by F1)")
    else:
        test_acc, test_precision, test_recall, test_f_score, test_iou, test_miou, test_mcc = matrix2index(cf_mtx)

    print("loss:      ", test_loss / (iteration + 1))
    print("acc:       {:.4f}".format(test_acc))
    print("precision: {:.4f}".format(test_precision))
    print("recall:    {:.4f}".format(test_recall))
    print("f1:        {:.4f}".format(test_f_score))
    print("iou:       {:.4f}".format(test_iou))
    print("miou:      {:.4f}".format(test_miou))
    print("mcc:       {:.4f}".format(test_mcc))
    print("[DEBUG] test_dataset.path =", getattr(test_dataset, "path", None))


if __name__ == "__main__":
    main()
