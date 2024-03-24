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

    
#class CropConcat(torch.nn.Module):
#    def __init__(self, crop=True):
#        super(CropConcat, self).__init__()
#        self.crop = crop

#    def do_crop(self, x, tw, th):
#        b, c, w, h = x.size()
#        x1 = int(round((w - tw) / 2.))
#        y1 = int(round((h - th) / 2.))
#        return x[:, :, x1:x1 + tw, y1:y1 + th]

#    def forward(self, x, y):
#        b, c, h, w = y.size()
#        if self.crop:
#            x = self.do_crop(x, h, w)
#        return torch.cat((x, y), dim=1)


class UnetResNet18(nn.Module):
    def __init__(self, output_channels):
        super(UnetResNet18, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        
        self.conv = nn.Sequential(self.base_model.conv1, self.base_model.bn1, self.base_model.relu) # first downsampling step
        self.maxpool = self.base_model.maxpool # second downsampling step
        self.encoder2 = self.base_model.layer1
        self.encoder3 = self.base_model.layer2
        self.encoder4 = self.base_model.layer3
        self.bottleneck = ConvBlock(256, 512)

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
        x1 = self.conv(x) # x1: 112
        x2 = self.maxpool(x1) # x2: 56
        x3 = self.encoder2(x2) # x3: 56
        x4 = self.encoder3(x3) # x4: 28
        x5 = self.encoder4(x4) # x5: 14
        
        x = self.bottleneck(x5) # x: 14

        x = self.upsample(x) # x: 28
        x = self.crop(x4, x) # x: 28
        x = self.conv_block6(x) # x: 28

        x = self.upsample(x) # x: 56
        x = self.crop(x3, x) # x: 56
        x = self.conv_block7(x) # x: 56

        x = self.upsample(x) # x: 112
        x = self.crop(x1, x) # x: 112
        x = self.conv_block8(x) # x: 112

        x = self.upsample(x) # x: 224
        #x = self.crop(x1, x)
        x = self.conv_block9(x) # x: 224

        x = self.last_conv(x) # x: 224

        return x
