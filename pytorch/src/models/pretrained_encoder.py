import torch
import torch.nn as nn
import torchvision

class pretrainedResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # load the pretrained resnet-50 model and freeze the weights 
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        
        # extract the first three layers of the resnet model
        self.conv1 = resnet.conv1 # layer 1: convolution with stride = 2 (output will have dims of 112 if input is 224)
        self.bn1 = resnet.bn1 
        self.relu = resnet.relu # layer 2: batch normalisation and relu (no change in dimensions)
        self.maxpool = resnet.maxpool # layer 3: max pooling with stride = 2 (output will have dims of 56 if input is 224)

        # resulting output dims for input dims kxk => (k/4)x(k/4) same level of downsampling as encoder trained from scratch

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    