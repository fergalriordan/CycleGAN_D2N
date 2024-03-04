# good learning rate for this model is 0.001

import torch
from torch.autograd import Variable


class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, batch_normalization=True):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.InstanceNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.InstanceNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.batch_normalization = batch_normalization

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)

        x = self.relu(x)

        return x


class DownSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(DownSample, self).__init__()
        self.down_sample = torch.nn.MaxPool2d(factor, factor)

    def forward(self, x):
        return self.down_sample(x)


class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = torch.nn.Upsample(scale_factor=factor, mode='bilinear')

    def forward(self, x):
        return self.up_sample(x)


class CropConcat(torch.nn.Module):
    def __init__(self, crop=True):
        super(CropConcat, self).__init__()
        self.crop = crop

    def do_crop(self, x, tw, th):
        b, c, w, h = x.size()
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return x[:, :, x1:x1 + tw, y1:y1 + th]

    def forward(self, x, y):
        b, c, h, w = y.size()
        if self.crop:
            x = self.do_crop(x, h, w)
        return torch.cat((x, y), dim=1)


class UpBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, batch_normalization=True, downsample=False):
        super(UpBlock, self).__init__()
        self.downsample = downsample
        self.conv = ConvBlock(input_channel, output_channel, batch_normalization=batch_normalization)
        self.downsampling = DownSample()

    def forward(self, x):
        x1 = self.conv(x)
        if self.downsample:
            x = self.downsampling(x1)
        else:
            x = x1
        return x, x1


class DownBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, batch_normalization=True, Upsample=False):
        super(DownBlock, self).__init__()
        self.Upsample = Upsample
        self.conv = ConvBlock(input_channel, output_channel, batch_normalization=batch_normalization)
        self.upsampling = UpSample()
        self.crop = CropConcat()

    def forward(self, x, y):
        if self.Upsample:
            x = self.upsampling(x)
        x = self.crop(y, x)
        x = self.conv(x)
        return x


class UNet(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UNet, self).__init__()
        # Down Blocks
        self.conv_block1 = ConvBlock(input_channel, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.conv_block4 = ConvBlock(128, 256)
        self.conv_block5 = ConvBlock(256, 512)

        # Up Blocks
        self.conv_block6 = ConvBlock(512 + 256, 256)
        self.conv_block7 = ConvBlock(256 + 128, 128)
        self.conv_block8 = ConvBlock(128 + 64, 64)
        self.conv_block9 = ConvBlock(64 + 32, 32)

        # Last convolution
        self.last_conv = torch.nn.Conv2d(32, output_channel, 1)

        self.crop = CropConcat()

        self.downsample = DownSample()
        self.upsample = UpSample()

    def forward(self, x):
        x1 = self.conv_block1(x)
        x = self.downsample(x1)
        x2 = self.conv_block2(x)
        x = self.downsample(x2)
        x3 = self.conv_block3(x)
        x = self.downsample(x3)
        x4 = self.conv_block4(x)
        x = self.downsample(x4)
        x5 = self.conv_block5(x)

        x = self.upsample(x5)
        x = self.crop(x4, x)
        x = self.conv_block6(x)

        x = self.upsample(x)
        x = self.crop(x3, x)
        x = self.conv_block7(x)

        x = self.upsample(x)
        x = self.crop(x2, x)
        x = self.conv_block8(x)

        x = self.upsample(x)
        x = self.crop(x1, x)
        x = self.conv_block9(x)

        x = self.last_conv(x)

        return x