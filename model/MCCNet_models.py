import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from vgg import VGG


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialAttention_no_s(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_no_s, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return x


class MCCM(nn.Module):
    def __init__(self, cur_channel):
        super(MCCM, self).__init__()
        self.relu = nn.ReLU(True)

        self.ca = ChannelAttention(cur_channel)
        self.sa_fg = SpatialAttention_no_s()
        self.sa_edge = SpatialAttention_no_s()
        self.sigmoid = nn.Sigmoid()
        self.FE_conv = BasicConv2d(cur_channel, cur_channel, 3, padding=1)
        self.BG_conv = BasicConv2d(cur_channel, cur_channel, 3, padding=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = BasicConv2d(cur_channel, cur_channel, 1)
        self.sa_ic = SpatialAttention()
        self.IC_conv = BasicConv2d(cur_channel, cur_channel, 3, padding=1)

        self.FE_B_I_conv = BasicConv2d(3 * cur_channel, cur_channel, 3, padding=1)

    def forward(self, x):
        x_ca = x.mul(self.ca(x))
        # Foreground attention
        x_sa_fg = self.sa_fg(x_ca)
        # Edge attention
        x_edge = self.sa_edge(x_ca)
        # Foreground and Edge (FE) feature
        x_fg_edge = self.FE_conv(x_ca.mul(self.sigmoid(x_sa_fg) + self.sigmoid(x_edge)))

        # Background feature
        x_bg = self.BG_conv(x_ca.mul(1 - self.sigmoid(x_sa_fg) - self.sigmoid(x_edge)))

        # Image-level content
        in_size = x.shape[2:]
        x_gap = self.conv1(self.global_avg_pool(x))
        x_up = F.interpolate(x_gap, size=in_size, mode="bilinear", align_corners=True)
        x_ic = self.IC_conv(x.mul(self.sa_ic(x_up)))

        x_RE_B_I = self.FE_B_I_conv(torch.cat((x_fg_edge, x_bg, x_ic), 1))

        return (x + x_RE_B_I), x_edge

class decoder(nn.Module):
    def __init__(self, channel=512):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder5 = nn.Sequential(
            BasicConv2d(channel, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(512, 512, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            BasicConv2d(1024, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 256, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 256, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            BasicConv2d(512, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 128, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1),
            BasicConv2d(128, 64, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            BasicConv2d(128, 64, 3, padding=1),
            BasicConv2d(64, 32, 3, padding=1),
        )
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

    def forward(self, x5, x4, x3, x2, x1):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        x5_up = self.decoder5(x5)
        s5 = self.S5(x5_up)

        x4_up = self.decoder4(torch.cat((x4, x5_up), 1))
        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        s1 = self.S1(x1_up)

        return s1, s2, s3, s4, s5


class MCCNet_VGG(nn.Module):
    def __init__(self, channel=32):
        super(MCCNet_VGG, self).__init__()
        # Backbone model
        self.vgg = VGG('rgb')

        self.MCCM5 = MCCM(512)
        self.MCCM4 = MCCM(512)
        self.MCCM3 = MCCM(256)
        self.MCCM2 = MCCM(128)
        self.MCCM1 = MCCM(64)

        self.decoder_rgb = decoder(512)

        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_rgb):
        x1_rgb = self.vgg.conv1(x_rgb)
        x2_rgb = self.vgg.conv2(x1_rgb)
        x3_rgb = self.vgg.conv3(x2_rgb)
        x4_rgb = self.vgg.conv4(x3_rgb)
        x5_rgb = self.vgg.conv5(x4_rgb)

        # LG means Local and Global, i.e., adjacent context information
        x5_MCCM, eg5 = self.MCCM5(x5_rgb)
        x4_MCCM, eg4 = self.MCCM4(x4_rgb)
        x3_MCCM, eg3 = self.MCCM3(x3_rgb)
        x2_MCCM, eg2 = self.MCCM2(x2_rgb)
        x1_MCCM, eg1 = self.MCCM1(x1_rgb)

        s1, s2, s3, s4, s5 = self.decoder_rgb(x5_MCCM, x4_MCCM, x3_MCCM, x2_MCCM, x1_MCCM)

        s3 = self.upsample2(s3)
        s4 = self.upsample4(s4)
        s5 = self.upsample8(s5)

        eg2 = self.upsample2(eg2)
        eg3 = self.upsample4(eg3)
        eg4 = self.upsample8(eg4)
        eg5 = self.upsample16(eg5)

        return s1, s2, s3, s4, s5, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4), self.sigmoid(
            s5), eg1, eg2, eg3, eg4, eg5
