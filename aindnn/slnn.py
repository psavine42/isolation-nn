import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, W=7, C=7, k=32):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        # pad first layer to 2 as in alphago
        self.conv1 = nn.Conv2d(C, k, 3, stride=1, padding=2, bias=False)
        #stack of layers with no bias and same size.
        self.conv2 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv3 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv4 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv42 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        # last layer is 1 x 1 filters as in alphago with bias
        self.conv5 = nn.Conv2d(k, W*W, 1, stride=1, bias=True)
        # self.fc1 = nn.Linear(1024, W*W)
        self.softmax = nn.Softmax()
        self._initialize_weights()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv42(x))
        x = self.relu(self.conv5(x))
        #print(x.size())
        logits = x.squeeze(-1).squeeze(-1)
        # in pytorch, Cross entropy includes a log_softmax
        # ~= log_softmax + Nll
        #logits = self.softmax(x)
        return logits


    # from official example - normals dist...
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class SLNet(nn.Module):
    def __init__(self, W=7, C=7, k=32):
        super(SLNet, self).__init__()

        self.relu = nn.ReLU()
        # pad first layer to 2 as in alphago
        self.conv1 = nn.Conv2d(C, k, 3, stride=1, padding=2, bias=False)
        #stack of layers with no bias and same size.
        self.conv2 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv3 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv4 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv42 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        # last layer is 1 x 1 filters as in alphago with bias
        self.conv5 = nn.Conv2d(k, W*W, 1, stride=1, bias=True)
        # self.fc1 = nn.Linear(1024, W*W)
        self.softmax = nn.Softmax()
        self._initialize_weights()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv42(x))
        x = self.relu(self.conv5(x)) # fail
        #print(x.size())
        logits = x.squeeze(-1).squeeze(-1)
        # Leaving softmax outside for now
        #logits = self.softmax(x)
        return logits


    # from official example - normals dist...
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PolicyNet(nn.Module):
    def __init__(self, sl_net, k):
        super(PolicyNet, self).__init__()
        # 1 x 1 filter over 49 channels, stride 1
        sl_net.conv5 = nn.Conv2d(k, 49, 1, stride=1, bias=True)
        self.baseNet = sl_net
        self.softmax = nn.Softmax()
        self.fc1 = nn.Linear(49, 256)
        self.fc2 = nn.Linear(256, 1)
        self.rewards = []
        self.saved_actions = []

    def forward(self, x):
        x = self.baseNet(x)
        x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.tanh(x)



class ValueNet(nn.Module):
    def __init__(self, sl_net):
        super(ValueNet, self).__init__()
        self.baseNet = sl_net
        

    def forward(self, x):
        outputs = self.baseNet(x)
        return F.softmax(outputs)
