""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class Reconstucter(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Reconstucter, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv_rec(n_channels, 16)
        self.down1 = Down_rec(16, 32)
        self.down2 = Down_rec(32, 64)
        self.down3 = Down_rec(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down_rec(128, 256 // factor)
        self.up1 = Up_rec(256, 128 // factor, bilinear)
        self.up2 = Up_rec(128, 64 // factor, bilinear)
        self.up3 = Up_rec(64, 32 // factor, bilinear)
        self.up4 = Up_rec(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu') 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Segmenter(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Segmenter, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu') 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Segmenter_32channel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Segmenter_32channel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu') 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Segmenter_3layer(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Segmenter_3layer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class Segmenter_lite(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Segmenter_lite, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 32

        self.inc = DoubleConv(n_channels, self.base_channel)
        self.down1 = Down(self.base_channel, self.base_channel * 2)
        self.down2 = Down(self.base_channel * 2, self.base_channel * 4)
        factor = 2 if bilinear else 1
        self.down3 = Down(self.base_channel * 4, self.base_channel * 8 // factor)
        self.up1 = Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear)
        self.up2 = Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear)
        self.up3 = Up(self.base_channel * 2, self.base_channel, bilinear)
        self.outc = OutConv(self.base_channel, n_classes)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits