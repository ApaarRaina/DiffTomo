import torch
import torch.nn as nn


class REDCNN(nn.Module):
    def __init__(self, load_pretrained=False, device='cpu'):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 96, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=0)

        self.deconv5 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.deconv4 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.deconv1 = nn.ConvTranspose2d(96, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=False)
        
        if load_pretrained:
            self.path = "weights/redcnn.pth"
            self.load_pretrained_weights(self.path, device=device)

    def forward(self, x):

        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))

        d5 = self.deconv5(x5)

        d5 = self.relu(d5 + x4)
        d4 = self.relu(self.deconv4(d5))

        d3 = self.deconv3(d4)

        d3 = self.relu(d3 + x2)
        d2 = self.relu(self.deconv2(d3))

        d1 = self.deconv1(d2)

        out = self.relu(d1 + x)

        return out

    def load_pretrained_weights(self, path, device='cpu'):
        state_dict = torch.load(path, map_location=device)

        self.load_state_dict(state_dict)