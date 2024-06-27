import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2)
    )

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.dconv_down1 = double_conv(3, 8)
        self.dconv_down2 = double_conv(8, 16)
        self.dconv_down3 = double_conv(16, 32)
        self.dconv_down4 = double_conv(32, 64)  # New level

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(32 + 64, 32)  # New level
        self.dconv_up2 = double_conv(16 + 32, 16)
        self.dconv_up1 = double_conv(8 + 16, 8)

        self.conv_last = nn.Conv2d(8, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)  # New level
        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)  # New level
        x = self.upsample(x)

        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)

        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

