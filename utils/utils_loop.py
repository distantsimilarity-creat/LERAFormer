import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils.utils_loss import build_loss, gen_matrix, matrix2index


def _best_threshold_from_counts(tp, fp, fn, tn, thr_list, select_by="f1", smooth=1e-7):
    tp = tp.float()
    fp = fp.float()
    fn = fn.float()
    tn = tn.float()

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou_fg = tp / (tp + fp + fn + smooth)

    iou_bg = tn / (tn + fp + fn + smooth)
    miou = (iou_bg + iou_fg) / 2.0

    denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + smooth)
    mcc = (tp * tn - fp * fn) / denom

    acc = (tp + tn) / (tp + tn + fp + fn + smooth)

    if select_by.lower() == "iou":
        score = iou_fg
    else:
        score = f1

    best_idx = int(torch.argmax(score).item())
    best_thr = float(thr_list[best_idx])

    metrics = {
        "thr": best_thr,
        "acc": float(acc[best_idx].item()),
        "precision": float(precision[best_idx].item()),
        "recall": float(recall[best_idx].item()),
        "f1": float(f1[best_idx].item()),
        "iou": float(iou_fg[best_idx].item()),
        "miou": float(miou[best_idx].item()),
        "mcc": float(mcc[best_idx].item()),
    }
    return best_thr, metrics


def loop_one_epoch(model, save_path, optimizer, lr_scheduler, epoch, end_epoch,
                   step_train, step_val, train_loader, val_loader, cuda, weights, num_classes,
                   use_amp: bool = True, grad_clip_norm: float = 1.0, stop_on_nonfinite: bool = True,
                   val_thr_list=None, save_best_thr: bool = True, select_best_thr_by: str = "f1",
                   ema=None, ema_eval: bool = True, ema_save: bool = True):
    device = next(model.parameters()).device
    use_amp = bool(use_amp and cuda)

    best_file = os.path.join(save_path, 'best_val_loss.txt')
    if os.path.exists(best_file):
        try:
            best_score = float(open(best_file, 'r', encoding='utf-8').read().strip())
        except Exception:
            best_score = float('inf')
    else:
        best_score = float('inf')

    train_loss = 0.0
    val_loss = 0.0

    scaler = GradScaler(enabled=use_amp)
    def _get_lr(opt):
        try:
            return max(float(pg.get("lr", 0.0)) for pg in opt.param_groups)
        except Exception:
            return float(opt.param_groups[0].get("lr", 0.0))

    epoch_lr = _get_lr(optimizer)



    def _isfinite(x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        return torch.isfinite(x).all()

    model.train()
    with tqdm(total=step_train, desc=f'Train {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= step_train:
                break

            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                imgs, labels, onehot_labels, indices = batch
            else:
                imgs, labels, onehot_labels = batch
                indices = None
            if cuda:
                imgs = imgs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                onehot_labels = onehot_labels.cuda(non_blocking=True)
                if weights is not None:
                    weights = weights.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(imgs)

            if indices is not None:
                bs = getattr(train_loader, "batch_sampler", None)
                if bs is not None and hasattr(bs, "update_batch"):
                    with torch.no_grad():
                        per_pix = F.cross_entropy(outputs.detach(), labels, reduction='none')
                        per_sample = per_pix.mean(dim=(1, 2))
                        bs.update_batch(indices, per_sample)

            if stop_on_nonfinite and (not _isfinite(outputs)):
                raise RuntimeError(f"[NonFinite outputs] epoch={epoch+1} iter={iteration+1}")

            loss = build_loss(outputs, labels, onehot_labels, weights=weights)

            if stop_on_nonfinite and (not torch.isfinite(loss)):
                raise RuntimeError(f"[NonFinite loss] epoch={epoch+1} iter={iteration+1}")

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()

            if ema is not None:
                ema.update(model)

            train_loss += float(loss.detach().item())
            epoch_lr = _get_lr(optimizer)
            pbar.set_postfix(**{'tra_loss': train_loss / (iteration + 1), 'lr': epoch_lr})
            pbar.update(1)

    ema_applied = False
    if ema is not None and ema_eval:
        try:
            ema.apply_to(model)
            ema_applied = True
        except Exception as _e:
            print(f"[EMA] apply_to failed: {_e}")
            ema_applied = False

    model.eval()

    cf_mtx = torch.zeros((num_classes, num_classes), device=device)

    do_thr_scan = (val_thr_list is not None) and (len(val_thr_list) > 0) and (num_classes == 2)
    if do_thr_scan:
        thr_list = torch.tensor(val_thr_list, device=device, dtype=torch.float32)
        T = thr_list.numel()
        tp = torch.zeros(T, device=device)
        fp = torch.zeros(T, device=device)
        fn = torch.zeros(T, device=device)
        tn = torch.zeros(T, device=device)

    with tqdm(total=step_val, desc=f'Valid {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= step_val:
                break
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                imgs, labels, onehot_labels, indices = batch
            else:
                imgs, labels, onehot_labels = batch
                indices = None
            with torch.no_grad():
                if cuda:
                    imgs = imgs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    onehot_labels = onehot_labels.cuda(non_blocking=True)
                    if weights is not None:
                        weights = weights.cuda(non_blocking=True)

                outputs = model(imgs)

            if stop_on_nonfinite and (not _isfinite(outputs)):
                raise RuntimeError(f"[NonFinite val outputs] epoch={epoch+1} iter={iteration+1}")

            loss = build_loss(outputs, labels, onehot_labels, weights=weights)
            if stop_on_nonfinite and (not torch.isfinite(loss)):
                raise RuntimeError(f"[NonFinite val loss] epoch={epoch+1} iter={iteration+1}")

            val_loss += float(loss.item())

            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            test_cf_mtx = gen_matrix(logits, labels, num_classes=num_classes, thr=None)[0]
            cf_mtx = cf_mtx + test_cf_mtx

            if do_thr_scan:
                prob_fg = torch.softmax(logits.float(), dim=1)[:, 1]
                valid = (labels >= 0) & (labels < num_classes)
                gt_v = labels[valid].long()
                pr_v = prob_fg[valid]
                for i, t in enumerate(thr_list):
                    pred1 = (pr_v > t)
                    gt1 = (gt_v == 1)
                    tp[i] += (pred1 & gt1).sum()
                    fp[i] += (pred1 & (~gt1)).sum()
                    fn[i] += ((~pred1) & gt1).sum()
                    tn[i] += ((~pred1) & (~gt1)).sum()

            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    val_acc, val_precision, val_recall, val_f_score, val_iou, val_miou, val_mcc = matrix2index(cf_mtx)

    best_thr = None
    best_thr_metrics = None
    if do_thr_scan:
        best_thr, best_thr_metrics = _best_threshold_from_counts(
            tp, fp, fn, tn, thr_list.tolist(), select_by=select_best_thr_by
        )
        print(f"[VAL THR] best_thr={best_thr:.2f}  "
              f"P={best_thr_metrics['precision']:.4f}  R={best_thr_metrics['recall']:.4f}  "
              f"F1={best_thr_metrics['f1']:.4f}  IoU={best_thr_metrics['iou']:.4f}")

        val_acc = best_thr_metrics['acc']
        val_precision = best_thr_metrics['precision']
        val_recall = best_thr_metrics['recall']
        val_f_score = best_thr_metrics['f1']
        val_iou = best_thr_metrics['iou']
        val_miou = best_thr_metrics['miou']
        val_mcc = best_thr_metrics['mcc']

        thr_epoch_path = os.path.join(save_path, f"thr_epoch_{epoch+1:03d}.txt")
        with open(thr_epoch_path, "w", encoding="utf-8") as f:
            f.write(str(best_thr))

    if 'ema_applied' in locals() and ema_applied:
        try:
            ema.restore(model)
        except Exception as _e:
            print(f"[EMA] restore failed: {_e}")

    checkpoint = {
        'model': model.state_dict(),
        'ema': (ema.state_dict() if (ema is not None and ema_save) else None),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else None,
        'epoch': epoch + 1,
    }
    torch.save(checkpoint, os.path.join(save_path, 'epc%03d-trloss%.3f-valoss%.3f-iou%.3f-f1-%.3f.pth' % (
        epoch + 1, train_loss / step_train, val_loss / step_val, val_iou, val_f_score)))

    this_val = val_loss / step_val

    if this_val < best_score:
        best_score = float(this_val)
        torch.save(checkpoint, os.path.join(save_path, 'best.pth'))
        with open(best_file, 'w', encoding='utf-8') as f:
            f.write(str(best_score))

    with open(os.path.join(save_path, 'log.txt'), 'a', encoding='utf-8') as f:
        line = (f'Epoch {epoch + 1}, Train Loss: {train_loss / step_train:.4f}, '
                f'Val Loss: {this_val:.4f}, Val IoU: {val_iou:.4f}, Val F1: {val_f_score:.4f}')
        if best_thr is not None and best_thr_metrics is not None:
            line += (f', BestThr: {best_thr:.2f}, '
                     f'BestThr_F1: {best_thr_metrics["f1"]:.4f}, BestThr_IoU: {best_thr_metrics["iou"]:.4f}')
        line += '\n'
        f.write(line)

    print('Epoch: ' + str(epoch + 1) + '/' + str(end_epoch) +
          ' >> train loss: %.3f || val loss: %.3f || val iou: %.3f || val f1: %.3f' %
          (train_loss / step_train, this_val, val_iou, val_f_score))

    return train_loss / step_train, this_val, val_iou, val_f_score
