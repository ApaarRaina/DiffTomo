import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2 if stride == 1 else 0
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=8, kernel_size=3, stride=2),
            ConvBlock(in_channels=8, out_channels=8, kernel_size=3, stride=1),

            ConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            ConvBlock(in_channels=16, out_channels=16, kernel_size=3, stride=1),

            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1),

            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1),

            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1),
        )

    def forward(self, x): 
               
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),

            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),

            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=1, stride=1),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),

            ConvBlock(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            ConvBlock(in_channels=16, out_channels=16, kernel_size=1, stride=1),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),

            ConvBlock(in_channels=8, out_channels=8, kernel_size=3, stride=1),
            ConvBlock(in_channels=8, out_channels=8, kernel_size=1, stride=1),

            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PriorModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            Encoder(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),

            Decoder(),
        )

    def forward(self, x):
        return self.net(x)