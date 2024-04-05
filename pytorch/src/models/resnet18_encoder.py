import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Replace all instances of BatchNorm2d with InstanceNorm2d
def replace_bn_with_in(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.InstanceNorm2d(child.num_features, affine=True))
        else:
            replace_bn_with_in(child)

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


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        self.base_model = models.resnet18(pretrained=True)

        replace_bn_with_in(self.base_model) # change the normalisation from batch to instance normalisation so that the normalisation is consistent across the entire network
        
        self.conv = nn.Sequential(self.base_model.conv1, self.base_model.bn1, self.base_model.relu) # first downsampling step
        self.maxpool = self.base_model.maxpool # second downsampling step
        self.encoder2 = self.base_model.layer1
        self.encoder3 = self.base_model.layer2
        self.encoder4 = self.base_model.layer3
        self.bottleneck = ConvBlock(256, 512)

    def forward(self, x):
        x1 = self.conv(x) # x1: 112
        x2 = self.maxpool(x1) # x2: 56
        x3 = self.encoder2(x2) # x3: 56
        x4 = self.encoder3(x3) # x4: 28
        x5 = self.encoder4(x4) # x5: 14
        x = self.bottleneck(x5) # x: 14

        return x, x4, x3, x1