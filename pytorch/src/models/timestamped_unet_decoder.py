import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, batch_normalization=True):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
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
    
class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = torch.nn.Upsample(scale_factor=factor, mode='bilinear')

    def forward(self, x):
        return self.up_sample(x)

class Timestamped_UNet_Decoder(torch.nn.Module):
    def __init__(
            self, encoder, timestamp_embedding_channels, output_channel
        ):
        super(Timestamped_UNet_Decoder, self).__init__()

        self.encoder = encoder
        self.timestamp_embedding_channels = timestamp_embedding_channels # define a number of channels for the timestamp embedding

        # Up Blocks
        self.conv_block6 = ConvBlock(512 + 256 + timestamp_embedding_channels, 256) # concatenate timestamp info with the output of the first upsampling step
        self.conv_block7 = ConvBlock(256 + 128, 128) # in the subsequent upsamplings, the timestamp information becomes merged with the spatial information
        self.conv_block8 = ConvBlock(128 + 64, 64)
        self.conv_block9 = ConvBlock(64 + 32, 32)

        # Last convolution
        self.last_conv = torch.nn.Conv2d(32, output_channel, 1)

        self.crop = CropConcat()
        self.upsample = UpSample()

    def forward(self, x, timestamp):
        x5, x4, x3, x2, x1 = self.encoder(x)

        # expand the timestamp info to the same height and width as the spatial info [B, C, H, W]
        # it can then be concatenated with the output of the first upsampling step
        timestamp = timestamp.view(-1, 1, 1, 1).expand(-1, self.timestamp_embedding_channels, x4.size(2), x4.size(3))

        x = self.upsample(x5)
        x = torch.cat((x, timestamp), dim=1) # concatenate timestamp info with spatial info
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