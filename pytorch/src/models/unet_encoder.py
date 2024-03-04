import torch

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


class UNet_Encoder(torch.nn.Module):
    def __init__(self, input_channel):
        super(UNet_Encoder, self).__init__()
        # Down Blocks
        self.conv_block1 = ConvBlock(input_channel, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.conv_block4 = ConvBlock(128, 256)
        self.conv_block5 = ConvBlock(256, 512)

        self.downsample = DownSample()
    
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

        return x5, x4, x3, x2, x1