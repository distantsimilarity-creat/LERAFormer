import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = min(window_size, input_resolution[0], input_resolution[1])
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        if self.shift_size >= self.window_size:
            self.shift_size = 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 128, patch_size: int = 4, in_chans: int = 14, embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        H, W = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.proj = nn.Linear(dim, dim // 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        x = self.proj(x)
        x = self.norm(x)
        C_out = x.shape[-1]
        x = x.view(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=self.dim_scale, mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C_out)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.output_dim = output_dim
        self.proj = nn.Linear(dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        x = self.proj(x)
        x = self.norm(x)
        x = x.view(B, H, W, self.output_dim).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, self.output_dim)
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        downsample: nn.Module = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim) if downsample is not None else None

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            x = blk(x)
        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x_out, x


class BasicLayerUp(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        upsample: nn.Module = None,
    ):
        super().__init__()
        self.upsample = upsample(input_resolution, dim=dim) if upsample is not None else None
        out_dim = dim // 2 if upsample is not None else dim
        out_resolution = (input_resolution[0] * 2, input_resolution[1] * 2) if upsample is not None else input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=out_resolution,
                num_heads=num_heads,
                window_size=min(window_size, out_resolution[0], out_resolution[1]),
                shift_size=0 if (i % 2 == 0) else min(window_size, out_resolution[0], out_resolution[1]) // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinUNetBaseline(nn.Module):
    def __init__(self, in_chans: int = 14, num_classes: int = 2, use_dger: bool = False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=128, patch_size=4, in_chans=in_chans, embed_dim=96)
        self.pos_drop = nn.Dropout(p=0.0)

        self.layers_down = nn.ModuleList([
            BasicLayer(dim=96,  input_resolution=(32, 32), depth=2, num_heads=3,  window_size=4, downsample=PatchMerging),
            BasicLayer(dim=192, input_resolution=(16, 16), depth=2, num_heads=6,  window_size=4, downsample=PatchMerging),
            BasicLayer(dim=384, input_resolution=(8, 8),   depth=2, num_heads=12, window_size=4, downsample=PatchMerging),
            BasicLayer(dim=768, input_resolution=(4, 4),   depth=2, num_heads=24, window_size=4, downsample=None),
        ])

        self.concat_linear_3 = nn.Linear(384 + 384, 384)
        self.concat_linear_2 = nn.Linear(192 + 192, 192)
        self.concat_linear_1 = nn.Linear(96 + 96, 96)

        self.layers_up = nn.ModuleList([
            BasicLayerUp(dim=768, input_resolution=(4, 4),   depth=2, num_heads=12, window_size=4, upsample=PatchExpand),
            BasicLayerUp(dim=384, input_resolution=(8, 8),   depth=2, num_heads=6,  window_size=4, upsample=PatchExpand),
            BasicLayerUp(dim=192, input_resolution=(16, 16), depth=2, num_heads=3,  window_size=4, upsample=PatchExpand),
        ])

        self.norm = nn.LayerNorm(96)
        self.final_up = FinalPatchExpand_X4(input_resolution=(32, 32), dim=96, output_dim=96)
        self.head = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor):
        x, _, _ = self.patch_embed(x)
        x = self.pos_drop(x)

        skips = []
        for layer in self.layers_down:
            x_out, x = layer(x)
            skips.append(x_out)
        return x, skips

    def forward_up_features(self, x: torch.Tensor, skips):
        x = self.layers_up[0](x)
        x = torch.cat([x, skips[2]], dim=-1)
        x = self.concat_linear_3(x)

        x = self.layers_up[1](x)
        x = torch.cat([x, skips[1]], dim=-1)
        x = self.concat_linear_2(x)

        x = self.layers_up[2](x)
        x = torch.cat([x, skips[0]], dim=-1)
        x = self.concat_linear_1(x)
        return x

    def up_x4(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.final_up(x)
        B, L, C = x.shape
        x = x.view(B, 128, 128, C).permute(0, 3, 1, 2).contiguous()
        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.forward_features(x)
        x = self.forward_up_features(x, skips)
        x = self.up_x4(x)
        return x


BEFLSFormer1_3 = SwinUNetBaseline


if __name__ == '__main__':
    torch.set_num_threads(1)
    model = SwinUNetBaseline(in_chans=14, num_classes=2,)
    x = torch.randn(1, 14, 128, 128)
    out = model(x)
    print('输出形状:', out.shape)