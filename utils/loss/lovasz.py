from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore:
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]
    return 100 * np.array(ious)

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:

        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero(as_tuple=False).squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def LovaszHingeLoss(inputs, target, num_classes = 2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs =  torch.softmax(inputs,1)[:,1:2,...].squeeze(1).contiguous()
    temp_target = target#.unsqueeze(1)

    BinaryLovas_loss  = lovasz_hinge(temp_inputs, temp_target, per_image=False)
    return BinaryLovas_loss

def LovaszSoftmaxLoss(inputs, target, num_classes=2, ignore_index=255,
                    classes='present', sanitize=True, clamp_val=30.0):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if (h != ht) or (w != wt):
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    logits = inputs.float()

    if sanitize:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
        logits = logits.clamp(min=-clamp_val, max=clamp_val)

    probas = torch.softmax(logits, dim=1)
    labels = target.long()

    loss = lovasz_softmax(probas, labels, classes=classes, per_image=False, ignore=ignore_index)

    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

    return loss


def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()


def entropy(prob):
    return -torch.sum(prob * torch.log(prob + 1e-10), dim=1, keepdim=True)


def adaptive_rectification_module(P, Y, num_classes):
    B, C, H, W = P.shape

    pseudo_label = torch.argmax(P, dim=1)
    M = (pseudo_label != Y).float().unsqueeze(1)

    Y_onehot = one_hot_encode(Y, num_classes)
    Ym = Y_onehot * M
    Pm = P * M

    mis_idx = pseudo_label.unsqueeze(1)
    pmis = torch.gather(Pm, 1, mis_idx)

    true_idx = Y.unsqueeze(1)
    ptru = torch.gather(Pm, 1, true_idx)

    lambda_a = 1.0 / (1 + pmis - ptru + 1e-10)

    ce = -torch.sum(Ym * torch.log(Pm + 1e-10), dim=1, keepdim=True)
    lambda_s = torch.exp(-ce)

    s = entropy(Pm)
    lambda_c = torch.exp(-s)

    lambda_total = lambda_a * lambda_s * lambda_c

    Pmr = lambda_total * Pm + (1 - lambda_total) * Ym

    Pc = P * (1 - M)
    Pr = Pmr + Pc

    return Pr


def rlcl_loss(P_cnn, P_trans, Y, num_classes=2):
    P_cnn_r = adaptive_rectification_module(P_cnn, Y, num_classes)
    P_trans_r = adaptive_rectification_module(P_trans, Y, num_classes)

    kl_cnn = F.kl_div(P_cnn.log_softmax(dim=1), P_trans_r, reduction='batchmean')
    kl_trans = F.kl_div(P_trans.log_softmax(dim=1), P_cnn_r, reduction='batchmean')

    return (kl_cnn + kl_trans) / 2.0