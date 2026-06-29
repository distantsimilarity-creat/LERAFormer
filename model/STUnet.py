import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用你项目中已经实现的 Swin 分层编码器（命名含 LSK，但该类本身是 Swin Encoder）
# 位置：model/LERAFormer.py 里定义了 LSK_SwinBodyEncoder，并输出四级特征 [64,32,16,8]
from model.LERAFormer import LSK_SwinBodyEncoder


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-2) != skip.size(-2) or x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class STUNet(nn.Module):
    def __init__(self, in_chans=14, num_classes=2,
                 embed_dim=56, depths=(2, 2, 6, 2), num_heads=(4, 8, 16, 32),
                 bilinear=True):
        super().__init__()
        self.encoder = LSK_SwinBodyEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads
        )
        self.up1 = UpBlock(in_ch=embed_dim * 8,  skip_ch=embed_dim * 4, out_ch=embed_dim * 4, bilinear=bilinear)
        self.up2 = UpBlock(in_ch=embed_dim * 4,  skip_ch=embed_dim * 2, out_ch=embed_dim * 2, bilinear=bilinear)
        self.up3 = UpBlock(in_ch=embed_dim * 2,  skip_ch=embed_dim * 1, out_ch=embed_dim * 1, bilinear=bilinear)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DoubleConv(embed_dim, embed_dim)
        )

        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def freeze_param(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        enc_skips = self.encoder(x)
        s1, s2, s3, s4 = enc_skips[0], enc_skips[1], enc_skips[2], enc_skips[3]

        x = self.up1(s4, s3)
        x = self.up2(x,  s2)
        x = self.up3(x,  s1)
        x = self.up4(x)

        out = self.head(x)
        return out


if __name__ == "__main__":
    model = STUNet(in_chans=14, num_classes=2)
    dummy = torch.randn(2, 14, 128, 128)
    y = model(dummy)
    print("output:", y.shape)
