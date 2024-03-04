import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UpSample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UpSample, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(input_channel, output_channel)

    def forward(self, x1, x2):
        x1 = self.up_sample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UnetResNet18(nn.Module):
    def __init__(self, output_channels):
        super(UnetResNet18, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        
        self.encoder1 = nn.Sequential(self.base_model.conv1, self.base_model.bn1, self.base_model.relu, self.base_model.maxpool)
        self.encoder2 = self.base_model.layer1
        self.encoder3 = self.base_model.layer2
        self.encoder4 = self.base_model.layer3

        self.bottleneck = ConvBlock(256, 256)
        self.up4 = UpSample(256 + 128, 128)
        self.up3 = UpSample(128 + 64, 64)
        self.up2 = UpSample(64 + 64, 32)
        self.up1 = ConvBlock(32 + 64, 16)  # Adjusted for concatenation
        
        # Adding two transposed convolution layers to upscale to 224x224
        #self.trans_conv1 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=1)
        #self.trans_conv2 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=1)

        # Alternative upsampling blocks: bilinear
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock(16, 16) # need conv blocks after the upsampling to emulate the learnability of transposed convolutions
        self.conv2 = ConvBlock(16, 16)
        
        self.final = nn.Conv2d(16, output_channels, kernel_size=1)

    def forward(self, x):
        
        #input_size = x.size()[2:] # store the input size for the final resizing step

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        x = self.bottleneck(x4)

        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(torch.cat([x, x1], dim=1))
        

        #x = F.relu(self.trans_conv1(x))
        #x = F.relu(self.trans_conv2(x))
        
        # Motivation for this is as follows:
        # The output needs to have the same dimensions as the input, otherwise the cycle consistency loss can't be calculated (and anyway, the fact that the output should match the input size is obvious)
        # In order to avoid checkerboard artifacts, the kernel size of the transposed convolutions must be a multiple of the stride
        # With these constraints, it is impossible to produce an output with the same dimensions as the input, given the training image size of 224x224
        # Therefore, we need to resize the output of the transposed convolution to match the input size (without this the images differ in size by a few pixels)
        #x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)


        # Alternative upsampling method: bilinear
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = self.conv2(x)

        return self.final(x)
