import torch
from torch import nn

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x, **kwargs):
        y = self.conv1(x)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.activation(y)
        y = self.conv3(y)
        y = self.activation(y)
        y = self.conv4(y)
        return y