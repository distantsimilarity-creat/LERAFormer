import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, stride=stride),
            ConvBNReLU(out_ch, out_ch, 3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpCat(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_chans: int = 14):
        super().__init__()
        self.stem = DoubleConv(in_chans, 64, stride=2)
        self.enc2 = DoubleConv(64, 128, stride=2)
        self.enc3 = DoubleConv(128, 256, stride=2)

    def forward(self, x: torch.Tensor):
        x1 = self.stem(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        return x1, x2, x3


class PatchEmbedFromFeature(nn.Module):
    def __init__(self, in_chans: int = 256, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, in_chans: int = 256, embed_dim: int = 256, depth: int = 4, num_heads: int = 8, drop: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbedFromFeature(in_chans=in_chans, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio=4.0, drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h, w = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class DecoderHead(nn.Module):
    def __init__(self, vit_ch: int = 256, skip3_ch: int = 256, skip2_ch: int = 128, skip1_ch: int = 64, num_classes: int = 2):
        super().__init__()
        self.up3 = UpCat(vit_ch, skip3_ch, 256)
        self.up2 = UpCat(256, skip2_ch, 128)
        self.up1 = UpCat(128, skip1_ch, 64)
        self.seg_head = nn.Sequential(
            ConvBNReLU(64, 32, 3),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, vit_feat: torch.Tensor, skip3: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
        x = self.up3(vit_feat, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 64 -> 128
        x = self.seg_head(x)
        return x


class TransUNetBaseline(nn.Module):
    def __init__(self, in_chans: int = 14, num_classes: int = 2, use_dger: bool = False):
        super().__init__()
        self.encoder = CNNEncoder(in_chans=in_chans)
        self.vit = ViTEncoder(in_chans=256, embed_dim=256, depth=4, num_heads=8, drop=0.0)
        self.decoder = DecoderHead(
            vit_ch=256,
            skip3_ch=256,
            skip2_ch=128,
            skip1_ch=64,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = self.encoder(x)
        vit_feat = self.vit(x3)
        out = self.decoder(vit_feat, x3, x2, x1)
        return out



if __name__ == '__main__':
    torch.set_num_threads(1)
    model = TransUNetBaseline(in_chans=14, num_classes=2,)
    x = torch.randn(1, 14, 128, 128)
    out = model(x)
    print('输出形状:', out.shape)
