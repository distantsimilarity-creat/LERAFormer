import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = None, groups: int = 1, act: bool = True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DWDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return self.sigmoid(attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attn)


class MSCA(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels, bias=False),
        ])
        self.bn = nn.BatchNorm2d(channels * 4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.bn(x)
        x = self.fuse(x)
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CNNEncoderLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, reduction: int):
        super().__init__()
        self.down = DWDownsample(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.msca = MSCA(out_ch, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.msca(x)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, drop: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PoolingTokenMixer(nn.Module):
    def __init__(self, pool_size: int = 3):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        feat = x.transpose(1, 2).reshape(b, c, h, w)
        mixed = self.pool(feat) - feat
        return mixed.flatten(2).transpose(1, 2)


class PTLBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = PoolingTokenMixer(pool_size=3)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.token_mixer(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class ETLBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int,
                 num_heads: int, sr_ratio: int, num_blocks: int, use_poolformer: bool = False):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_ch, out_ch, kernel_size, stride, padding)
        blocks = []
        for _ in range(num_blocks):
            if use_poolformer:
                blocks.append(PTLBlock(out_ch))
            else:
                blocks.append(ETLBlock(out_ch, num_heads=num_heads, sr_ratio=sr_ratio))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h, w = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, h, w)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, h, w)
        return x

class CrossEncoderFusion(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNAct(ch * 2, ch, kernel_size=1, padding=0),
            ConvBNAct(ch, ch, kernel_size=3)
        )

    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        if cnn_feat.shape[-2:] != trans_feat.shape[-2:]:
            trans_feat = F.interpolate(trans_feat, size=cnn_feat.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([cnn_feat, trans_feat], dim=1)
        return self.fuse(x)


class SegFormerDecoder(nn.Module):
    def __init__(self, in_channels: List[int], embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            for ch in in_channels
        ])
        self.fuse = nn.Sequential(
            ConvBNAct(embed_dim * 4, embed_dim, kernel_size=1, padding=0),
            ConvBNAct(embed_dim, embed_dim, kernel_size=3),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        target_size = feats[0].shape[-2:]
        outs = []
        for feat, proj in zip(feats, self.proj):
            x = proj(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        x = self.head(x)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return x

class LEFormerBaseline(nn.Module):
    def __init__(self, in_chans: int = 14, num_classes: int = 2,):
        super().__init__()

        cnn_channels = [32, 64, 160, 192]
        reductions = [8, 4, 2, 1]
        kernels = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        paddings = [3, 1, 1, 1]

        self.cnn_layers = nn.ModuleList([
            CNNEncoderLayer(in_chans, cnn_channels[0], kernels[0], strides[0], paddings[0], reductions[0]),
            CNNEncoderLayer(cnn_channels[0], cnn_channels[1], kernels[1], strides[1], paddings[1], reductions[1]),
            CNNEncoderLayer(cnn_channels[1], cnn_channels[2], kernels[2], strides[2], paddings[2], reductions[2]),
            CNNEncoderLayer(cnn_channels[2], cnn_channels[3], kernels[3], strides[3], paddings[3], reductions[3]),
        ])

        self.trans_layers = nn.ModuleList([
            TransformerEncoderLayer(in_chans, cnn_channels[0], kernels[0], strides[0], paddings[0], num_heads=1, sr_ratio=8, num_blocks=2, use_poolformer=True),
            TransformerEncoderLayer(cnn_channels[0], cnn_channels[1], kernels[1], strides[1], paddings[1], num_heads=2, sr_ratio=4, num_blocks=2, use_poolformer=False),
            TransformerEncoderLayer(cnn_channels[1], cnn_channels[2], kernels[2], strides[2], paddings[2], num_heads=5, sr_ratio=2, num_blocks=2, use_poolformer=False),
            TransformerEncoderLayer(cnn_channels[2], cnn_channels[3], kernels[3], strides[3], paddings[3], num_heads=6, sr_ratio=1, num_blocks=3, use_poolformer=False),
        ])

        self.cef = nn.ModuleList([CrossEncoderFusion(ch) for ch in cnn_channels])
        self.decoder = SegFormerDecoder(in_channels=cnn_channels, embed_dim=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        cnn_feats = []
        trans_feats = []

        xc = x
        xt = x
        for cnn_layer, trans_layer in zip(self.cnn_layers, self.trans_layers):
            xc = cnn_layer(xc)
            xt = trans_layer(xt)
            cnn_feats.append(xc)
            trans_feats.append(xt)

        fused_feats = [f(c, t) for f, c, t in zip(self.cef, cnn_feats, trans_feats)]
        out = self.decoder(fused_feats, out_size=out_size)
        return out

if __name__ == '__main__':
    torch.set_num_threads(1)
    model = LEFormerBaseline(in_chans=14, num_classes=2,)
    x = torch.randn(1, 14, 128, 128)
    y = model(x)
    print('输出形状:', y.shape)  # [1, 2, 128, 128]
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'参数量: {total_params:.2f} M')
    print('LEFormer-style baseline 已成功运行')