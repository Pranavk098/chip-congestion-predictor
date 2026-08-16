"""Small U-Net for predicting DRC-violation-likelihood heatmaps from pre-route layout rasters."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if odd input sizes cause off-by-one mismatch
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 'Attention U-Net', 2018).

    Gates the encoder skip connection with the coarser decoder feature map
    before concatenation, so the network learns to suppress skip-connection
    regions irrelevant to the (rare, localized) violation target instead of
    passing every encoder activation through unfiltered. Standard component
    in recent DRC-hotspot literature's enhanced U-Net backbones (e.g. MAGNet's
    Dynamic Attention Module)."""

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.w_gate(gate)
        s = self.w_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = nn.functional.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        attn = self.psi(self.relu(g + s))
        return skip * attn


class AttentionUp(nn.Module):
    """Same as Up, but the skip connection passes through an AttentionGate
    (gated by the pre-upsample decoder feature map) before concatenation."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate(gate_ch=in_ch, skip_ch=skip_ch, inter_ch=max(skip_ch // 2, 8))
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        gated_skip = self.attn(x, skip)
        x = self.up(x)
        dy = gated_skip.size(2) - x.size(2)
        dx = gated_skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([gated_skip, x], dim=1)
        return self.conv(x)


class CongestionUNet(nn.Module):
    """
    Input:  (B, in_channels, H, W) layout rasters (cell density, pin density, RUDY, ...)
    Output: (B, 1, H, W) logits for DRC-violation-likelihood per bin (apply sigmoid for probability)
    """

    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        self.inc = ConvBlock(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.up1 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


class AttentionCongestionUNet(nn.Module):
    """CongestionUNet with attention gates on every skip connection. Same
    depth/width as CongestionUNet so the two are a clean apples-to-apples
    ablation (see model/train.py --arch)."""

    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        self.inc = ConvBlock(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.up1 = AttentionUp(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = AttentionUp(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3 = AttentionUp(base_ch * 2, base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def build_model(arch, in_channels, base_ch=32):
    if arch == "unet":
        return CongestionUNet(in_channels=in_channels, base_ch=base_ch)
    if arch == "attention_unet":
        return AttentionCongestionUNet(in_channels=in_channels, base_ch=base_ch)
    raise ValueError(f"unknown arch: {arch}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for arch in ["unet", "attention_unet"]:
        m = build_model(arch, in_channels=3, base_ch=32)
        x = torch.randn(2, 3, 64, 64)
        y = m(x)
        print(f"{arch}: output shape {y.shape}, params {count_params(m)}")
