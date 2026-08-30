# coding:utf-8
# LiEAF-Net: Lightweight Multi-Scale Elevation-Aware Fusion Network
# Author: Furkat Sultonov

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ====================== Base Building Blocks ====================== #

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

def ConvMixer_Block(in_dim, out_dim, kernel_size=3, padding=1):
    return nn.Sequential(
        Residual(nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, groups=in_dim, padding=padding),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )),
        nn.Conv2d(in_dim, out_dim, 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True)
    )

# ====================== EADASK Fusion Module ====================== #

class EADASK(nn.Module):
    """
    Elevation-Aware Double-Attention Selective Kernel (EADASK) fusion module.

    1. Cross-modal calibration (SE-style gating between RGB and nDSM streams)
    2. Modality-specific multi-scale encoding (1x1, 3x3, 5x5, 7x7 receptive fields)
    3. Spatial attention over scale branches
    4. Channel attention over scale branches
    """
    def __init__(self, in_ch, out_ch, heavy=False, r=16, L=32):
        super().__init__()
        self.heavy = heavy

        if heavy:
            mid_ch = out_ch // 4  # Each branch outputs this
            d = max(int(out_ch / r), L)

            # === Modality-specific branches ===
            # RGB branches
            self.rgb_1x1 = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True)
            )
            self.rgb_3x3 = ConvMixer_Block(in_ch, mid_ch)

            # nDSM (elevation) branches
            self.depth_1x1 = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True)
            )
            self.depth_3x3 = ConvMixer_Block(in_ch, mid_ch)

            # Shared cascaded convolutions (for efficiency)
            self.conv_5x5 = ConvMixer_Block(mid_ch, mid_ch)
            self.conv_7x7 = ConvMixer_Block(mid_ch, mid_ch)

            # === Cross-modal calibration ===
            self.cross_modal_rgb = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // 4, in_ch, 1, bias=False),
                nn.Sigmoid()
            )
            self.cross_modal_depth = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // 4, in_ch, 1, bias=False),
                nn.Sigmoid()
            )

            # === Dual Attention ===
            # Spatial Attention
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(1, 4, 1, bias=False),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True)
            )

            # Channel Attention
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.channel_fc = nn.Sequential(
                nn.Conv2d(mid_ch, d, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True)
            )
            self.channel_attn = nn.Conv2d(d, out_ch, 1)

            # Output projection
            self.fc_out = nn.Conv2d(mid_ch, out_ch, 1)

        else:
            # Lightweight fusion: simple separate projection + concatenation
            self.rgb_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch // 2, 1, bias=False),
                nn.BatchNorm2d(out_ch // 2),
                nn.ReLU(inplace=True)
            )
            self.depth_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch // 2, 1, bias=False),
                nn.BatchNorm2d(out_ch // 2),
                nn.ReLU(inplace=True)
            )

    def forward(self, rgb, depth):
        if self.heavy:
            # === Step 1: Cross-modal calibration ===
            rgb_gate = self.cross_modal_rgb(depth)
            depth_gate = self.cross_modal_depth(rgb)
            rgb_calibrated = rgb * (1 + rgb_gate)
            depth_calibrated = depth * (1 + depth_gate)

            # === Step 2: Extract modality-specific multi-scale features ===
            # RGB path
            rgb_1x1 = self.rgb_1x1(rgb_calibrated)
            rgb_3x3 = self.rgb_3x3(rgb_calibrated)
            rgb_5x5 = self.conv_5x5(rgb_3x3)
            rgb_7x7 = self.conv_7x7(rgb_5x5)

            # nDSM path
            depth_1x1 = self.depth_1x1(depth_calibrated)
            depth_3x3 = self.depth_3x3(depth_calibrated)
            depth_5x5 = self.conv_5x5(depth_3x3)
            depth_7x7 = self.conv_7x7(depth_5x5)

            # === Step 3: Combine multi-scale features ===
            # Average RGB and nDSM features per scale for compact fusion
            x_1x1 = (rgb_1x1 + depth_1x1) / 2
            x_3x3 = (rgb_3x3 + depth_3x3) / 2
            x_5x5 = (rgb_5x5 + depth_5x5) / 2
            x_7x7 = (rgb_7x7 + depth_7x7) / 2

            # Stack for dual attention
            U_stacked = torch.stack([x_1x1, x_3x3, x_5x5, x_7x7], dim=1)  # [B, 4, C//4, H, W]
            feats_U = torch.sum(U_stacked, dim=1)  # [B, C//4, H, W]

            # === Step 4: Spatial Attention ===
            feats_S_spatial, _ = torch.max(feats_U, dim=1, keepdim=True)  # [B, 1, H, W]
            feats_Z_spatial = self.spatial_attn(feats_S_spatial).unsqueeze(2)  # [B, 4, 1, H, W]
            attn_spatial = F.softmax(feats_Z_spatial, dim=1)

            V_spatial = U_stacked * attn_spatial
            feats_V_spatial = torch.sum(V_spatial, dim=1)
            spatial_out = feats_V_spatial + feats_U  # Residual connection

            # === Step 5: Channel Attention ===
            feats_S_channel = self.gap(spatial_out)
            feats_Z_channel = self.channel_fc(feats_S_channel)
            attn_channel = self.channel_attn(feats_Z_channel)
            attn_channel = attn_channel.view(attn_channel.size(0), 4, attn_channel.size(1)//4, 1, 1)
            attn_channel = F.softmax(attn_channel, dim=1)

            V_channel = V_spatial * attn_channel
            feats_V_channel = torch.sum(V_channel, dim=1)
            channel_out = feats_V_channel + spatial_out  # Residual connection

            # === Step 6: Output projection ===
            output = self.fc_out(channel_out)
            return output
        else:
            # Lightweight path: process separately and concatenate
            rgb_feat = self.rgb_conv(rgb)
            depth_feat = self.depth_conv(depth)
            return torch.cat([rgb_feat, depth_feat], dim=1)


# ====================== Lightweight ASPP ====================== #

class LightweightASPP(nn.Module):
    def __init__(self, in_ch, out_ch, atrous_rates=[3,6,9,12]):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=r, dilation=r, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for r in atrous_rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        total_ch = out_ch * (len(atrous_rates)+2)
        self.project = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = [self.conv1x1(x)]
        res.extend([conv(x) for conv in self.atrous_convs])
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(gp)
        return self.project(torch.cat(res, dim=1))


# ====================== Lightweight Decoder with DS + SE ====================== #

class DecoderBlockDSSE(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, reduction=4):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_ch, out_ch, 1, bias=False)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//reduction, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        skip = self.skip_conv(skip)
        x = self.dw_conv(x) + skip
        x = x * self.se(x)
        return x


# ====================== Multi-scale Attention SegHead ====================== #

class MultiScaleSegHeadAttn(nn.Module):
    def __init__(self, low_ch, mid_ch, high_ch, num_classes):
        super().__init__()
        self.low_conv = nn.Conv2d(low_ch, 16, 3, padding=1, bias=False)
        self.mid_conv = nn.Conv2d(mid_ch, 16, 3, padding=1, bias=False)
        self.high_conv = nn.Conv2d(high_ch, 16, 3, padding=1, bias=False)
        self.attn = nn.Sequential(
            nn.Conv2d(48, 48, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Conv2d(48, num_classes, 1)

    def forward(self, low, mid, high):
        low_f = self.low_conv(low)
        mid_f = self.mid_conv(F.interpolate(mid, size=low.shape[2:], mode='bilinear', align_corners=True))
        high_f = self.high_conv(F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=True))
        feat = torch.cat([low_f, mid_f, high_f], dim=1)
        feat = feat * self.attn(feat)
        return self.classifier(feat)


# ====================== LiEAF-Net ====================== #

class LiEAFNet(nn.Module):
    """
    LiEAF-Net: Lightweight Multi-Scale Elevation-Aware Fusion Network.

    Args:
        n_class: Number of segmentation classes
        aux_loss: Whether to use auxiliary loss
    """
    def __init__(self, n_class=6, aux_loss=False):
        super().__init__()
        self.aux_loss = aux_loss

        # Pretrained MobileNetV3-Large
        mobilenet_rgb = models.mobilenet_v3_large(pretrained=True)
        mobilenet_elevation = models.mobilenet_v3_large(pretrained=True)

        rgb_features = list(mobilenet_rgb.features)
        elevation_features = list(mobilenet_elevation.features)
        first_conv = elevation_features[0][0]

        # Modify first conv for single-channel nDSM input
        new_first_conv = nn.Conv2d(1, first_conv.out_channels,
                                   kernel_size=first_conv.kernel_size,
                                   stride=first_conv.stride,
                                   padding=first_conv.padding,
                                   bias=False)
        nn.init.kaiming_normal_(new_first_conv.weight, mode='fan_out', nonlinearity='relu')
        elevation_features[0][0] = new_first_conv

        # Encoder stages (MobileNetV3-Large)
        self.rgb_stem = nn.Sequential(*rgb_features[:1])       # 16 channels
        self.rgb_stage1 = nn.Sequential(*rgb_features[1:4])    # 24 channels
        self.rgb_stage2 = nn.Sequential(*rgb_features[4:7])    # 40 channels
        self.rgb_stage3 = nn.Sequential(*rgb_features[7:13])   # 112 channels
        self.rgb_stage4 = nn.Sequential(*rgb_features[13:16])  # 160 channels

        # nDSM (elevation) encoder stages
        self.elevation_stem = nn.Sequential(*elevation_features[:1])
        self.elevation_stage1 = nn.Sequential(*elevation_features[1:4])
        self.elevation_stage2 = nn.Sequential(*elevation_features[4:7])
        self.elevation_stage3 = nn.Sequential(*elevation_features[7:13])
        self.elevation_stage4 = nn.Sequential(*elevation_features[13:16])

        # Fusion modules: lightweight fusion at stages 0-1, EADASK at stages 2-4
        self.fusion_stem = EADASK(16, 16, heavy=False)
        self.fusion1 = EADASK(24, 24, heavy=False)
        self.fusion2 = EADASK(40, 40, heavy=True)
        self.fusion3 = EADASK(112, 112, heavy=True)
        self.fusion4 = EADASK(160, 160, heavy=True)

        # Context module (LASPP)
        self.context = LightweightASPP(160, 256, atrous_rates=[3,6,9,12])

        # Decoder blocks
        self.dec3 = DecoderBlockDSSE(256, 112, 128)
        self.dec2 = DecoderBlockDSSE(128, 40, 64)
        self.dec1 = DecoderBlockDSSE(64, 24, 32)
        self.dec0 = DecoderBlockDSSE(32, 16, 16)

        # Multi-scale SegHead
        self.seg_head = MultiScaleSegHeadAttn(16, 32, 64, n_class)

        # Auxiliary head for deep supervision
        if aux_loss:
            self.aux_head = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, n_class, 1)
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 4, H, W] where first 3 channels are RGB, last channel is nDSM

        Returns:
            If training with aux_loss: (main_output, aux_output)
            Otherwise: main_output
        """
        B, _, H, W = x.shape
        rgb, depth = x[:, :3], x[:, 3:]

        # === Encoder: Extract hierarchical features ===
        # Stem stage
        rgb0 = self.rgb_stem(rgb)
        d0 = self.elevation_stem(depth)
        fuse0 = self.fusion_stem(rgb0, d0)

        # Stage 1
        rgb1 = self.rgb_stage1(rgb0)
        d1 = self.elevation_stage1(d0)
        fuse1 = self.fusion1(rgb1, d1)

        # Stage 2
        rgb2 = self.rgb_stage2(rgb1)
        d2 = self.elevation_stage2(d1)
        fuse2 = self.fusion2(rgb2, d2)

        # Stage 3
        rgb3 = self.rgb_stage3(rgb2)
        d3 = self.elevation_stage3(d2)
        fuse3 = self.fusion3(rgb3, d3)

        # Stage 4
        rgb4 = self.rgb_stage4(rgb3)
        d4 = self.elevation_stage4(d3)
        fuse4 = self.fusion4(rgb4, d4)

        # === Context Module ===
        context_feat = self.context(fuse4)

        # === Decoder: Progressive upsampling with skip connections ===
        d3 = self.dec3(context_feat, fuse3)
        d2 = self.dec2(d3, fuse2)
        d1 = self.dec1(d2, fuse1)
        d0 = self.dec0(d1, fuse0)

        # === Segmentation Head ===
        seg_out = self.seg_head(d0, d1, d2)
        seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=True)

        # === Auxiliary Output (for training) ===
        if self.aux_loss and self.training:
            aux_out = self.aux_head(d2)
            aux_out = F.interpolate(aux_out, size=(H, W), mode='bilinear', align_corners=True)
            return seg_out, aux_out

        return seg_out
