import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from einops import rearrange
from collections import OrderedDict

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_c=14, embed_dim=56, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            attn += mask
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = x.view(B, H, W, C)
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = attn_windows.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shifted_x[:, :H, :W, :].contiguous().view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class LSK_SwinBodyEncoder(nn.Module):
    def __init__(self, in_chans=14, embed_dim=56, depths=(2, 2, 6, 2), num_heads=(4, 8, 16, 32)):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=2, in_c=in_chans, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=0.)
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            dim = embed_dim * (2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads[i_layer],
                    window_size=8,
                    shift_size=0 if (i % 2 == 0) else 4,
                    drop_path=dpr[sum(depths[:i_layer]) + i]
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            if i_layer < len(depths) - 1:
                self.layers.append(nn.Linear(4 * dim, 2 * dim))

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for blk in layer:
                    x = blk(x, H, W)
                skips.append(x.view(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2))
            else:
                B, L, C = x.shape
                x = x.view(B, H, W, C)
                x0 = x[:, 0::2, 0::2, :]
                x1 = x[:, 1::2, 0::2, :]
                x2 = x[:, 0::2, 1::2, :]
                x3 = x[:, 1::2, 1::2, :]
                x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
                x = layer(x)
                H, W = H // 2, W // 2
        return skips

class PiDiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(14, 56, kernel_size=3, padding=1),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(56, 112, kernel_size=3, padding=1),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(112, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(224, 448, kernel_size=3, padding=1),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        skips = []
        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                skips.append(x)
        return skips

class Channel_Exchange_Attention(nn.Module):
    def __init__(self, channel, ratio=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, body_feat, edge_feat):
        b, c, _, _ = body_feat.size()
        y_b = self.avg_pool(body_feat).view(b, c)
        y_b = self.fc(y_b).view(b, c, 1, 1)
        trans2edge = body_feat * y_b.expand_as(body_feat)
        y_e = self.avg_pool(edge_feat).view(b, c)
        y_e = self.fc(y_e).view(b, c, 1, 1)
        trans2body = edge_feat * y_e.expand_as(edge_feat)

        return trans2edge, trans2body

class LCAF(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, edge, body):
        B, C, H, W = edge.shape
        edge = edge.flatten(2).transpose(1, 2)
        body = body.flatten(2).transpose(1, 2)
        edge = self.norm(edge)
        body = self.norm(body)
        q = self.proj_q(edge)
        k = self.proj_k(body)
        v = self.proj_v(body)
        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        win_size = self.window_size
        pad_h = (win_size - H % win_size) % win_size
        pad_w = (win_size - W % win_size) % win_size
        q = F.pad(q.permute(0, 2, 3, 1).reshape(B, H, W, C), (0, 0, 0, pad_w, 0, pad_h))
        k = F.pad(k.permute(0, 2, 3, 1).reshape(B, H, W, C), (0, 0, 0, pad_w, 0, pad_h))
        v = F.pad(v.permute(0, 2, 3, 1).reshape(B, H, W, C), (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = q.shape
        q = q.reshape(B, Hp // win_size, win_size, Wp // win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, win_size * win_size, C)
        k = k.reshape(B, Hp // win_size, win_size, Wp // win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, win_size * win_size, C)
        v = v.reshape(B, Hp // win_size, win_size, Wp // win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, win_size * win_size, C)
        q = q.reshape(-1, win_size * win_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(-1, win_size * win_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(-1, win_size * win_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, win_size * win_size, C)
        out = out.reshape(B, Hp // win_size, Wp // win_size, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        out = out[:, :H, :W, :].reshape(B, H * W, C)
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out + edge.transpose(1, 2).reshape(B, C, H, W)

class LERAFormer(nn.Module):
    def __init__(self, in_chans=14, num_classes=2, use_dger=False):
        super().__init__()
        self.use_dger = use_dger
        self.body_encoder = LSK_SwinBodyEncoder(in_chans=in_chans)
        self.edge_encoder = PiDiNet()
        chans = [56, 112, 224, 448]
        self.lsk_body = nn.ModuleList([LSKblock(dim=c) for c in chans])
        self.lsk_edge = nn.ModuleList([LSKblock(dim=c) for c in chans])
        self.lcafs = nn.ModuleList([
            LCAF(56), LCAF(112), LCAF(224), LCAF(448),
        ])
        chans = [56, 112, 224, 448]
        self.cfca = nn.ModuleList([
            Channel_Exchange_Attention(channel=chans[i], ratio=2) for i in range(4)
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(56, 28, kernel_size=2, stride=2),
                nn.BatchNorm2d(28), nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(112, 56, kernel_size=2, stride=2),
                nn.BatchNorm2d(56), nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(224, 112, kernel_size=2, stride=2),
                nn.BatchNorm2d(112), nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(448, 224, kernel_size=2, stride=2),
                nn.BatchNorm2d(224), nn.ReLU(inplace=True),
            ),
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(28 + 56 + 112 + 224, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        body_skips = self.body_encoder(x)
        edge_skips = self.edge_encoder(x)

        fused = []
        for i in range(4):
            body_feat = body_skips[i]
            edge_feat = edge_skips[i]

            body_feat = self.lsk_body[i](body_feat)
            edge_feat = self.lsk_edge[i](edge_feat)
            if self.use_dger:
                with torch.no_grad():
                    pseudo_body = torch.argmax(body_feat, dim=1)
                    pseudo_edge = torch.argmax(edge_feat, dim=1)
                    disagree = (pseudo_body != pseudo_edge).float()
                    disagree = disagree.unsqueeze(1)
                edge_reprogrammed = edge_feat * (disagree + 0.1 * (1 - disagree))

            else:
                edge_reprogrammed = edge_feat

            trans2cnn, cnn2trans = self.cfca[i](body_feat, edge_reprogrammed)

            edge_input = trans2cnn + edge_reprogrammed
            body_input = cnn2trans + body_feat

            fused_feat = self.lcafs[i](edge_input, body_input)
            fused.append(fused_feat)
        up = []
        for i in range(4):
            up_i = self.decoder[i](fused[i])
            up.append(F.interpolate(up_i, size=(128, 128), mode='bilinear', align_corners=True))

        out = torch.cat(up, dim=1)
        out = self.final_conv(out)
        return out

if __name__ == '__main__':
    model = LERAFormer(in_chans=14, num_classes=2, use_dger=True)
    x = torch.randn(1, 14, 128, 128)
    out = model(x)
    print("输出形状:", out.shape)
    print("DGER 已启用，边缘分支已成功被不一致区域重编程！")