"""
LandslideNet-Seg (PyTorch, single-file)
--------------------------------------
Goal: provide a LandslideNet-style baseline that can be trained with a *semantic segmentation* pipeline
(i.e., forward() returns logits [B, num_classes, H, W]) and supports multi-channel remote-sensing inputs.

Notes:
- The LandslideNet paper is YOLOv8-based and mentions several modules (LSAWC, CWA, Adaptive-DCN, DSConv, DBB).
  Public official code may be incomplete/unreleased; this file provides a practical, dependency-light approximation:
  * LSAWC: group-conv branch + spatial-attention branch
  * CWA  : compound attention (channel + spatial + coordinate attention)
  * Adaptive-DCN: attention-gated conv (deformable conv is attempted if torchvision ops exist; otherwise fallback)
  * DSConv: lightweight "snake-like" large-kernel depthwise separable conv (pure PyTorch)
  * DBB  : multi-branch conv block (train-time multi-branch; optional re-param is NOT implemented here)

- This implementation is designed to be robust: it runs with only torch installed.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# utils
# ----------------------------
def _make_divisible(v: int, divisor: int = 8) -> int:
    return int(math.ceil(v / divisor) * divisor)


def autopad(k: int, p: Optional[int] = None, d: int = 1) -> int:
    # same-padding for odd kernel sizes
    if p is None:
        p = (k - 1) // 2 * d
    return p


# ----------------------------
# basic layers (YOLO-like)
# ----------------------------
class Conv(nn.Module):
    """Conv2d + BN + SiLU."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None,
                 g: int = 1, d: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise separable conv (dw + pw)."""
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1, act=act)
        self.pw = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SPPF(nn.Module):
    """SPPF block from YOLO: fast spatial pyramid pooling."""
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# ----------------------------
# attention blocks: Spatial / Channel / Coordinate
# ----------------------------
class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,1,H,W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.bn(self.conv(attn)))
        return attn


class SEAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        mid = max(8, c // r)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return w


class CoordAtt(nn.Module):
    """
    Coordinate Attention (simplified): encodes H and W separately then generates two attention maps.
    Reference idea: "Coordinate Attention for Efficient Mobile Network Design" (Hou et al.)
    """
    def __init__(self, c: int, r: int = 32):
        super().__init__()
        mid = max(8, c // r)
        self.conv1 = nn.Conv2d(c, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, c, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, c, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)  # (B,C,W,1)
        y = torch.cat([x_h, x_w], dim=2)  # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w).permute(0, 1, 3, 2))
        return a_h * a_w


class CWA(nn.Module):
    """
    Compound Weight Attention (CWA) approximation:
    Combine channel attention (SE), spatial attention, and coordinate attention.
    """
    def __init__(self, c: int, se_r: int = 16):
        super().__init__()
        self.se = SEAttention(c, r=se_r)
        self.sa = SpatialAttention(kernel_size=7)
        self.ca = CoordAtt(c, r=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_c = self.se(x)                      # (B,C,1,1)
        a_s = self.sa(x)                      # (B,1,H,W)
        a_xy = self.ca(x)                     # (B,C,H,W)
        # normalize-ish combination (keep stable)
        out = x * a_xy
        out = out * a_c
        out = out * a_s
        return out


class LSAWC(nn.Module):
    """
    LSAWC (Local Spatial Attention Weight Compensation) approximation:
    - branch1: group convolution for lightweight local feature extraction
    - branch2: spatial attention map
    - fuse: feature * attention + residual
    """
    def __init__(self, c: int, groups: int = 4):
        super().__init__()
        g = min(groups, c)
        while c % g != 0 and g > 1:
            g -= 1
        self.feat = nn.Sequential(
            Conv(c, c, k=3, s=1, g=g),
            Conv(c, c, k=1, s=1),
        )
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feat(x)
        a = self.sa(x)
        return x + f * a


# ----------------------------
# DSConv-lite & Adaptive-DCN-lite
# ----------------------------
class DSConvLite(nn.Module):
    """
    Lightweight approximation of DSConv:
    - large-kernel depthwise separable conv in both axes (1xk and kx1)
    - gated residual
    This keeps the "elongated structure sensitivity" spirit while staying dependency-free.
    """
    def __init__(self, c: int, k: int = 7):
        super().__init__()
        assert k % 2 == 1, "k should be odd"
        self.dw1 = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False)
        self.dw2 = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw1(x) + self.dw2(x)
        y = self.act(self.bn(y))
        y = self.act(self.bn2(self.pw(y)))
        g = self.gate(x)
        return x + y * g


class AdaptiveDCN(nn.Module):
    """
    Adaptive-DCN approximation:
    - tries to use torchvision deform_conv2d if available (offset-only).
    - otherwise uses a standard 3x3 conv, but gated by CWA to simulate adaptiveness.
    """
    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.k = k
        self.attn = CWA(c)
        self.offset = nn.Conv2d(c, 2 * k * k, 3, padding=1)
        self.use_deform = False

        # fallback conv
        self.fallback = Conv(c, c, k=k, s=1)

        # optional deform conv
        try:
            import torchvision  # noqa
            from torchvision.ops import deform_conv2d  # type: ignore
            self.deform_conv2d = deform_conv2d
            self.weight = nn.Parameter(torch.empty(c, c, k, k))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = None
            self.use_deform = True
        except Exception:
            self.deform_conv2d = None
            self.weight = None
            self.bias = None
            self.use_deform = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn = self.attn(x)
        if self.use_deform and self.deform_conv2d is not None and self.weight is not None:
            # offsets predicted from attentive features
            offset = self.offset(x_attn)
            # torchvision deform_conv2d supports offset-only deformation
            return self.deform_conv2d(x_attn, offset, self.weight, self.bias, stride=1, padding=autopad(self.k))
        # fallback: standard conv on attentive features
        return self.fallback(x_attn)


# ----------------------------
# DBB (multi-branch conv)
# ----------------------------
class DBBConv(nn.Module):
    """
    Diverse Branch Block (DBB) inspired multi-branch conv (training form).
    No re-parameterization is applied here; this is sufficient for a baseline.
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        p = autopad(k)
        # Branch A: kxk conv
        self.b1 = Conv(c1, c2, k=k, s=s, p=p, act=False)
        # Branch B: 1x1 -> kxk
        self.b2 = nn.Sequential(
            Conv(c1, c2, k=1, s=s, p=0, act=False),
            Conv(c2, c2, k=k, s=1, p=p, act=False),
        )
        # Branch C: avgpool -> 1x1
        self.b3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=k, stride=s, padding=p),
            Conv(c1, c2, k=1, s=1, p=0, act=False),
        )
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b1(x) + self.b2(x) + self.b3(x)
        return self.act(y)


# ----------------------------
# YOLOv8-style bottleneck and C2f
# ----------------------------
class Bottleneck(nn.Module):
    def __init__(self, c: int, shortcut: bool = True,
                 use_dcn: bool = False, use_dsconv: bool = False):
        super().__init__()
        self.cv1 = Conv(c, c, 1, 1)
        if use_dcn:
            self.cv2 = AdaptiveDCN(c, k=3)
        elif use_dsconv:
            self.cv2 = DSConvLite(c, k=7)
        else:
            self.cv2 = Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """
    Simplified C2f: split channels, then apply n bottlenecks on one split, concat and fuse.
    """
    def __init__(self, c1: int, c2: int, n: int = 2,
                 use_dcn: bool = False, use_dsconv: bool = False):
        super().__init__()
        self.c = c2
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.ModuleList([Bottleneck(c2, shortcut=True, use_dcn=use_dcn, use_dsconv=use_dsconv) for _ in range(n)])
        self.cv3 = Conv(c2 * (n + 2), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        outs = [y1, y2]
        for block in self.m:
            y2 = block(y2)
            outs.append(y2)
        return self.cv3(torch.cat(outs, dim=1))


# ----------------------------
# LandslideNet-Seg (adapted)
# ----------------------------
class LandslideNetSeg(nn.Module):
    """
    A practical LandslideNet-style baseline for semantic segmentation on 128x128 patches.
    forward(x): logits [B, num_classes, H, W]
    """
    def __init__(self,
                 in_chans: int = 14,
                 num_classes: int = 2,
                 width: float = 0.75,
                 depth: float = 0.67):
        super().__init__()

        # channel plan (kept close to YOLO-ish scaling, but small enough for segmentation training)
        c1 = _make_divisible(int(32 * width))
        c2 = _make_divisible(int(64 * width))
        c3 = _make_divisible(int(128 * width))
        c4 = _make_divisible(int(256 * width))

        n1 = max(1, int(round(2 * depth)))
        n2 = max(1, int(round(2 * depth)))
        n3 = max(1, int(round(2 * depth)))

        # backbone
        self.stem = Conv(in_chans, c1, 3, 2)                 # 128 -> 64
        self.lsawc = LSAWC(c1)

        self.down1 = Conv(c1, c2, 3, 2)                      # 64 -> 32
        self.c2f1 = C2f(c2, c2, n=n1, use_dsconv=True)        # DSConv spirit on mid-level

        self.down2 = Conv(c2, c3, 3, 2)                      # 32 -> 16
        self.c2f2 = C2f(c3, c3, n=n2, use_dcn=True)           # Adaptive-DCN spirit on deeper

        self.down3 = Conv(c3, c4, 3, 2)                      # 16 -> 8
        self.c2f3 = C2f(c4, c4, n=n3, use_dcn=True)
        self.sppf = SPPF(c4, c4)

        # neck (FPN-ish) with DBB blocks
        self.lat5 = Conv(c4, c3, 1, 1)
        self.lat4 = Conv(c3, c3, 1, 1)
        self.lat3 = Conv(c2, c3, 1, 1)

        self.smooth4 = DBBConv(c3, c3, 3, 1)
        self.smooth3 = DBBConv(c3, c3, 3, 1)

        # decoder to full resolution
        self.up1 = nn.Sequential(
            DBBConv(c3, c2, 3, 1),
        )
        self.up2 = nn.Sequential(
            DBBConv(c2, c1, 3, 1),
        )
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # backbone
        x = self.stem(x)         # 64
        x = self.lsawc(x)

        c3 = self.c2f1(self.down1(x))     # 32
        c4 = self.c2f2(self.down2(c3))    # 16
        c5 = self.sppf(self.c2f3(self.down3(c4)))  # 8

        # neck (top-down)
        p5 = self.lat5(c5)  # -> c3 channels (8)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")  # 16
        p4 = self.smooth4(p4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")  # 32
        p3 = self.smooth3(p3)

        # decoder (32->64->128) then resize to input size just in case
        y = F.interpolate(p3, scale_factor=2.0, mode="bilinear", align_corners=False)  # 64
        y = self.up1(y)
        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)  # 128
        y = self.up2(y)
        y = self.head(y)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return y


if __name__ == "__main__":
    # quick sanity test
    model = LandslideNetSeg(in_chans=14, num_classes=2)
    x = torch.randn(2, 14, 128, 128)
    y = model(x)
    print("out:", y.shape)  # [2, 2, 128, 128]
