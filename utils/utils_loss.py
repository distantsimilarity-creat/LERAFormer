import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss.lovasz import LovaszSoftmaxLoss


def boundary_loss(pred_prob, gt, smooth=1e-5):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred_prob.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred_prob.device).view(1, 1, 3, 3)

    pred_dx = F.conv2d(pred_prob, sobel_x, padding=1)
    pred_dy = F.conv2d(pred_prob, sobel_y, padding=1)
    pred_bound = torch.sqrt(pred_dx ** 2 + pred_dy ** 2 + smooth)

    gt_dx = F.conv2d(gt, sobel_x, padding=1)
    gt_dy = F.conv2d(gt, sobel_y, padding=1)
    gt_bound = torch.sqrt(gt_dx ** 2 + gt_dy ** 2 + smooth)

    return torch.mean(torch.abs(pred_bound - gt_bound))


def build_loss(outputs, labels, onehot_labels=None, weights=None,
              clamp_val=30.0, eps=1e-6, loss_cfg=None, **kwargs):
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    logits = logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(-clamp_val, clamp_val)

    if weights is None and loss_cfg is not None:
        cls_w = None
        if isinstance(loss_cfg, dict):
            cls_w = loss_cfg.get("cls_weights", None)
        else:
            cls_w = getattr(loss_cfg, "cls_weights", None)
        if cls_w is not None:
            weights = torch.tensor(cls_w, dtype=torch.float32)
    if weights is not None:
        weights = weights.to(logits.device, dtype=torch.float32)
    ce = nn.CrossEntropyLoss(weight=weights, ignore_index=255)(logits, labels)

    prob = torch.softmax(logits, dim=1)
    p_fg = prob[:, 1:2, ...]
    gt_fg = (labels == 1).float().unsqueeze(1)

    inter = (p_fg * gt_fg).sum(dim=(0, 2, 3))
    denom = p_fg.sum(dim=(0, 2, 3)) + gt_fg.sum(dim=(0, 2, 3)) + eps
    dice = 1.0 - (2.0 * inter / denom).mean()

    alpha, gamma = 0.75, 4.0
    focal = -alpha * (1 - p_fg) ** gamma * gt_fg * torch.log(p_fg + eps)
    focal = focal.mean()

    lovasz = LovaszSoftmaxLoss(logits, labels, num_classes=2)

    bound = boundary_loss(p_fg, gt_fg)

    total = 0.2 * ce + 0.25 * dice + 0.25 * focal + 0.2 * lovasz + 0.1 * bound

    if not torch.isfinite(total):
        total = ce if torch.isfinite(ce) else torch.zeros((), device=logits.device, dtype=logits.dtype)

    return total

def gen_matrix(pred_logits, gt_mask, num_classes=2, thr=None):
    logits = pred_logits[0] if isinstance(pred_logits, (tuple, list)) else pred_logits

    if thr is None:
        pred = torch.argmax(logits, dim=1)
    else:
        prob_fg = torch.softmax(logits.float(), dim=1)[:, 1]
        pred = (prob_fg > float(thr)).long()

    mask = (gt_mask >= 0) & (gt_mask < num_classes)
    count = torch.bincount(num_classes * gt_mask[mask].int() + pred[mask].int(), minlength=num_classes ** 2)

    cf_mtx = count.reshape(num_classes, num_classes)
    oa, precision, recall, f_score, iou, miou, mcc = matrix2index(cf_mtx)

    return cf_mtx, oa, precision, recall, f_score, iou, miou, mcc


def matrix2index(matrix, smooth=1e-3):
    cf_mtx = matrix
    smooth = torch.tensor(smooth, dtype=torch.float32).to(cf_mtx.device)
    ts1 = torch.tensor(1, dtype=torch.float32).to(cf_mtx.device)

    accuracy = torch.div(torch.sum(torch.diagonal(cf_mtx)), torch.maximum(torch.sum(cf_mtx), ts1))
    precision = torch.div(torch.diagonal(cf_mtx), torch.maximum(cf_mtx.sum(0), ts1))
    recall = torch.div(torch.diagonal(cf_mtx), torch.maximum(cf_mtx.sum(1), ts1))
    f_score = torch.div(2 * precision * recall, (precision + recall + smooth))

    miou = torch.div(torch.diag(cf_mtx),
                     (torch.sum(cf_mtx, axis=1) + torch.sum(cf_mtx, axis=0) - torch.diag(cf_mtx) + smooth))
    mcc = torch.div(
        (torch.prod(torch.diagonal(cf_mtx)) - torch.prod(torch.diagonal(torch.fliplr(cf_mtx))) + smooth),
        (torch.sqrt(
            torch.prod(cf_mtx.sum(0), dtype=torch.float32) * torch.prod(cf_mtx.sum(1), dtype=torch.float32)) + smooth)
    )

    oa = accuracy.item()
    precision = precision[1].item()
    recall = recall[1].item()
    f_score = f_score[1].item()
    iou = miou[1].item()
    miou = torch.mean(miou).item()
    mcc = mcc.item()

    return oa, precision, recall, f_score, iou, miou, mcc
