import torch
import torch.nn as nn
import torch.nn.functional


"""
Feature Planes -> encode a move history

"""

"""
Modules

Conv Block
    conv 256 filters, 3 x 3, stride 1
    Batch Norm
    Relu

Residual Block
1 layer of resnt 

"""


class ConvBlock(nn.Module):
    def __init__(self, channels, filters=256, size=3, stride=1, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(channels, filters, size, stride=stride, groups=groups) #, padding=1)
        self.bnorm = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bnorm(x)
        return self.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, filters=256, size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.filter = filters
        self.conv1 = nn.Conv2d(channels, self.filter, size,
                               stride=stride, padding=1)
        self.conv2 = nn.Conv2d(self.filter, self.filter, size,
                               stride=stride, padding=1)
        self.bnorm = nn.BatchNorm2d(self.filter)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bnorm(x)
        x += inputs
        out = self.relu(x)
        return out


class ResidualCatBlock(nn.Module):
    def __init__(self, channels, filters=256, size=2, stride=1):
        super(ResidualCatBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, filters //2, size,
                               stride=stride, padding=0)
        self.conv2 = nn.Conv2d(filters//2, filters //2, size,
                               stride=stride, padding=1)
        self.bnorm = nn.BatchNorm2d(filters //2)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs
        print('\n----res-in', inputs.size())
        x = self.conv1(inputs)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bnorm(x)
        print('to_cat', residual.size(), x.size())

        residual.downsample(2)
        merged = torch.cat([residual, x], 1)
        return self.relu(merged)


class PolicyHead(nn.Module):
    def __init__(self, channels=64, filters=2, size=1, stride=1):
        super(PolicyHead, self).__init__()
        self.conv_block = ConvBlock(channels, filters=filters, size=size, groups=1)
        self.fc = nn.Linear(50, 50)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        x = self.conv_block(inputs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.softmax(x)
        return out


class ValueHead(nn.Module):
    def __init__(self, channels=64, hidden_size=64, filters=1, size=1, stride=1):
        super(ValueHead, self).__init__()
        self.conv_block = ConvBlock(channels, filters=filters, size=size, groups=1)
        self.hidden = nn.Linear(25, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        x = self.conv_block(inputs)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.fc(x)
        return self.tanh(x)


