import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size//2, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x

class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=14, embed_dims=[64, 128, 320, 512]):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(in_chans, embed_dims[0], patch_size=7, stride=4)
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads=1,  sr_ratio=8) for _ in range(3)
        ])

        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2)
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads=2,  sr_ratio=4) for _ in range(4)
        ])

        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2)
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads=5,  sr_ratio=2) for _ in range(18)
        ])

        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2)
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads=8,  sr_ratio=1) for _ in range(3)
        ])

        self.norm = nn.LayerNorm(embed_dims[-1])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

class SegFormerHead(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], embed_dim=256, num_classes=2):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(c, embed_dim, 1) for c in in_channels
        ])
        self.fuse = nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False)
        self.pred = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        x0 = F.interpolate(self.proj[0](features[0]), size=features[0].shape[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(self.proj[1](features[1]), size=features[0].shape[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.proj[2](features[2]), size=features[0].shape[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.proj[3](features[3]), size=features[0].shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.fuse(x)
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        x = self.pred(x)
        return x

class SegFormer(nn.Module):
    def __init__(self, in_chans=14, num_classes=2):
        super().__init__()
        self.backbone = MixVisionTransformer(in_chans=in_chans)
        self.decode_head = SegFormerHead(num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.decode_head(feats)
        return out