import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.instance_n1 = nn.InstanceNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.instance_n2 = nn.InstanceNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.instance_n1(self.conv1(x)))
        x = self.relu(self.instance_n2(self.conv2(x)))
        return x

class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = torch.nn.Upsample(scale_factor=factor, mode='bilinear')

    def forward(self, x):
        return self.up_sample(x)
    
class CropConcat(torch.nn.Module):
    def __init__(self):
        super(CropConcat, self).__init__()

    def forward(self, x, y):
        # Adjust x to match the size of y
        _, _, h_y, w_y = y.size()
        x = F.interpolate(x, size=(h_y, w_y), mode='bilinear', align_corners=False)
        
        return torch.cat((x, y), dim=1)

class TimestampedResNet18Decoder(nn.Module):
    def __init__(self, encoder, timestamp_embedding_channels, output_channels):
        super(TimestampedResNet18Decoder, self).__init__()

        self.encoder = encoder
        self.timestamp_embedding_channels = timestamp_embedding_channels # define a number of channels for the timestamp embedding

        # Up Blocks
        self.conv_block6 = ConvBlock(512 + 128, 256)
        self.conv_block7 = ConvBlock(256 + 64, 128)
        self.conv_block8 = ConvBlock(128 + 64, 64)
        self.conv_block9 = ConvBlock(64, 32)

        self.crop = CropConcat()
        self.upsample = UpSample()

        # Last convolution
        self.last_conv = torch.nn.Conv2d(32, 3, 1)

    def forward(self, x):

        x, x4, x3, x1 = self.encoder(x)

        # expand the timestamp info to the same height and width as the spatial info [B, C, H, W]
        # it can then be concatenated with the output of the first upsampling step
        timestamp = timestamp.view(-1, 1, 1, 1).expand(-1, self.timestamp_embedding_channels, x4.size(2), x4.size(3))

        x = self.upsample(x) # x: 28
        x = torch.cat((x, timestamp), dim=1) # concatenate timestamp info with spatial info
        x = self.crop(x4, x) # x: 28
        x = self.conv_block6(x) # x: 28

        x = self.upsample(x) # x: 56
        x = self.crop(x3, x) # x: 56
        x = self.conv_block7(x) # x: 56

        x = self.upsample(x) # x: 112
        x = self.crop(x1, x) # x: 112
        x = self.conv_block8(x) # x: 112

        x = self.upsample(x) # x: 224
        x = self.conv_block9(x) # x: 224

        x = self.last_conv(x) # x: 224

        return x