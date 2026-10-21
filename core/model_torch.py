"""
PyTorch Reference Architecture of the Two-Stage GPR Reflection Enhancer.
Decouples Instantaneous Phase Manifold Extraction (Stage 1) and Dilated Spatial Reasoning (Stage 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MiniPhaseUNet(nn.Module):
    def __init__(self, in_ch=9, out_ch=2):
        super().__init__()
        self.inc = DoubleConv(in_ch, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(64 + 32, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(32 + 16, 16)
        self.outc = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.conv1(torch.cat([self.up1(x3), x2], dim=1))
        x = self.conv2(torch.cat([self.up2(x), x1], dim=1))
        return self.outc(x)


class HilbertLayer(nn.Module):
    """
    Computes analytic signal z(t) = s(t) + i*H[s(t)] along the depth/time axis (dim=2).
    Directly runs on MPS/CUDA/CPU using torch.fft.
    """
    def forward(self, x):
        x_fft = torch.fft.fft(x, dim=2)
        H = x.shape[2]
        h = torch.zeros_like(x_fft)
        h[:, :, 0, :] = x_fft[:, :, 0, :]
        h[:, :, 1 : (H + 1) // 2, :] = 2 * x_fft[:, :, 1 : (H + 1) // 2, :]
        z = torch.fft.ifft(h, dim=2)
        return z, torch.abs(z)


class AdaptiveWaveletDiff(nn.Module):
    def __init__(self, scales=4, k=7, in_ch=1):
        super().__init__()
        self.scales = scales
        g = scales if in_ch == scales else 1
        self.diff = nn.Conv2d(
            in_ch, scales, (9, 1), padding=(4, 0), groups=g, bias=False
        )
        self.smooth = nn.Conv2d(
            scales, scales, (k, 1), padding=(k // 2, 0), groups=scales, bias=False
        )
        t = torch.linspace(-4, 4, 9)
        for s in range(scales):
            self.diff.weight.data[s, 0, :, 0] = -t * torch.exp(
                -(t**2) / (2 * (0.5 * 1.5**s) ** 2)
            )
            gk = torch.exp(-(torch.linspace(-2, 2, k) ** 2))
            self.smooth.weight.data[s, 0, :, 0] = gk / gk.sum()

    def forward(self, x, w):
        dev = self.diff.weight.device
        x = x.to(dev)
        w = w.to(dev)
        raw = self.diff(x)
        we = w.repeat(1, self.scales, 1, 1) if w.shape[1] == 1 else w
        return self.smooth(raw * we) / (self.smooth(we) + 1e-6)


class PhaseManifoldProcessor(nn.Module):
    def __init__(self, scales=4, smooth_kernel_size=3):
        super().__init__()
        self.s = scales
        self.hilbert = HilbertLayer()
        self.omega_extractor = AdaptiveWaveletDiff(self.s, smooth_kernel_size, 1)
        self.chirp_extractor = AdaptiveWaveletDiff(self.s, smooth_kernel_size, self.s)

        total_feats = 1 + self.s * 2
        self.feat_norm = nn.BatchNorm2d(total_feats)
        self.proj = MiniPhaseUNet(in_ch=total_feats, out_ch=2)

    def forward(self, x):
        z, A = self.hilbert(x)
        zr, zi = z.real.float(), z.imag.float()

        dzr, dzi = self.omega_extractor.diff(zr), self.omega_extractor.diff(zi)
        num = dzi * zr.repeat(1, self.s, 1, 1) - dzr * zi.repeat(1, self.s, 1, 1)
        omega = self.omega_extractor.smooth(
            (num / (A.pow(2).repeat(1, self.s, 1, 1) + 1e-6))
            * A.repeat(1, self.s, 1, 1)
        ) / (self.omega_extractor.smooth(A.repeat(1, self.s, 1, 1)) + 1e-6)
        chirp = self.chirp_extractor(omega, A)

        raw_feats = torch.cat([A, omega, chirp], dim=1)
        norm_feats = self.feat_norm(raw_feats)
        return z, self.proj(norm_feats)


class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GlobalContextBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 4
        self.b1 = DilatedConvBlock(in_ch, mid_ch, dilation=1, padding=1)
        self.b2 = DilatedConvBlock(in_ch, mid_ch, dilation=2, padding=2)
        self.b3 = DilatedConvBlock(in_ch, mid_ch, dilation=4, padding=4)
        self.b4 = DilatedConvBlock(in_ch, mid_ch, dilation=16, padding=16)
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fusion(
            torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        )


class Stage2_GlobalContextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(4, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.bottleneck = GlobalContextBottleneck(128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(128 + 64, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(64 + 32, 32)
        self.outc = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x_neck = self.bottleneck(x3)
        x = self.conv1(torch.cat([self.up1(x_neck), x2], dim=1))
        x = self.conv2(torch.cat([self.up2(x), x1], dim=1))
        return self.outc(x)


class TwoStageGPRNet(nn.Module):
    """
    Two-stage network architecture:
    Stage 1: Phase Manifold Extraction
    Stage 2: Dilated Spatial Reasoning & Signed Polarity Probability Extraction
    """
    def __init__(self, scales=4, smooth_kernel_size=3):
        super().__init__()
        self.stage1 = PhaseManifoldProcessor(scales=scales, smooth_kernel_size=smooth_kernel_size)
        self.stage2 = Stage2_GlobalContextNet()

    def forward(self, x):
        z, p1_logits = self.stage1(x)
        p1_prob = torch.sigmoid(p1_logits)

        zr, zi = z.real.float(), z.imag.float()
        amp = torch.sqrt(zr**2 + zi**2) + 1e-8

        # Stage 2 Input: [zr/A, zi/A, p1_prob_pos, p1_prob_neg] -> 4 channels
        stage2_in = torch.cat([zr / amp, zi / amp, p1_prob], dim=1)
        p2_logits = self.stage2(stage2_in)
        return p1_logits, p2_logits
