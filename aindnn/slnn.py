import torch
import torch.nn as nn
import math, os
import torch.nn.functional as F
import aindnn.layers as layers
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, W=7, C=7, k=32):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        # pad first layer to 2 as in alphago
        self.conv1 = nn.Conv2d(C, k, 3, stride=1, padding=2, bias=False)
        # stack of layers with no bias and same size.
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
        return x

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
    def __init__(self, W=7, C=7, k=32, chkpt=None):
        super(SLNet, self).__init__()

        self.relu = nn.ReLU()
        # pad first layer to 2 as in alphago
        self.conv1 = nn.Conv2d(C, k, 3, stride=1, padding=2, bias=False)
        # stack of layers with no bias and same size.
        self.conv2 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv3 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv4 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        self.conv42 = nn.Conv2d(k, k, 3, stride=1, bias=False)
        # last layer is 1 x 1 filters as in alphago with bias
        self.conv5 = nn.Conv2d(k, W*W, 1, stride=1, bias=True)
        # self.fc1 = nn.Linear(1024, W*W)
        self.softmax = nn.Softmax()
        if chkpt:
            self.load_weights(chkpt)
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv42(x))
        x = self.relu(self.conv5(x)) # fail
        logits = x.squeeze(-1).squeeze(-1)
        # Leaving softmax outside for now
        return logits

    def load_weights(self, chkpt):
        pretrained_dict = torch.load(chkpt)
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)

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
    def __init__(self, sl_net, k=7):
        super(PolicyNet, self).__init__()
        # 1 x 1 filter over 49 channels, stride 1
        self.baseNet = sl_net
        self.conv_last = nn.Conv2d(64, 64, 1, stride=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)
        self.rewards = []
        self.saved_actions = []

    def forward(self, x):
        x = self.baseNet(x)
        x = self.conv_last(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.softmax(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.tanh(x)


class ValueNet(nn.Module):
    def __init__(self, sl_net):
        super(ValueNet, self).__init__()
        self.baseNet = sl_net

    def forward(self, x):
        outputs = self.baseNet(x)
        return F.softmax(outputs)


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def load_checkpoint(self):
        pass

    def forward(self, x):
        outputs = self.baseNet(x)
        return F.softmax(outputs)


class ZeroNet(nn.Module):
    """
    todo documentation
    """
    def __init__(self,
                 channels=7,
                 n_blocks=7,
                 filters=64,
                 verbose=False,
                 checkpoint=None,
                 **kwdargs):
        """

        :param channels:
        :param n_blocks:
        :param filters:
        :param verbose:
        :param kwdargs:
        """
        super(ZeroNet, self).__init__()
        self.channels = channels
        self.blocks = n_blocks
        self.filters = filters
        self.verbose = verbose
        self.checkpoint = checkpoint
        # nn
        self.first_conv = layers.ConvBlock(self.channels, filters=self.filters)
        self.tower = nn.Sequential(
            OrderedDict(
                [('mod_{}'.format(i), layers.ResidualBlock(self.filters, filters=self.filters)) \
                 for i in range(self.blocks)]
            )
        )
        self.policy = layers.PolicyHead(channels=self.filters,
                                        filters=2,
                                        size=1,
                                        stride=1)
        self.value = layers.ValueHead(channels=self.filters,
                                      filters=1,
                                      size=1,
                                      stride=1)
        if self.checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, chkpt):
        if not os.path.exists(chkpt):
            print('ERROR - CHECKPOINT FILE DOES NOT EXIST AT: {}'.format(chkpt))
            return
        pretrained_dict = torch.load(chkpt)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)

    def save_checkpoint(self, chkpt):
        torch.save(self.state_dict(), chkpt)

    def log(self, *msg):
        if self.verbose is True:
            print(*msg)

    def forward(self, inputs):
        conv_out = self.first_conv(inputs)
        self.log('conv_out', conv_out.size())

        tower_out = self.tower(conv_out)
        self.log('tower_out', tower_out.size())

        policy = self.policy(tower_out)
        self.log('policy_OUT', policy.size())

        value = self.value(tower_out)
        self.log('value_OUT', value.size())
        return policy, value





