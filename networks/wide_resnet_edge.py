import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from canny_net import CannyNet
import sys
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet_Edge(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_Edge, self).__init__()
        self.in_planes = 16
        self.edge_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.canny = CannyNet(threshold=1.8, use_cuda=True, requires_grad=False)
        self.conv1 = conv3x3(3, nStages[0])
        self.conv4 = conv3x3(4, nStages[0])
        # self.conv2 = conv3x3(1, nStages[0])
        # self.conv3 = conv3x3(320, nStages[1])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        # self.layer1_edge = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, is_edge=True)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        # self.layer2_edge = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, is_edge=True)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, is_edge=False):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        if is_edge:
            for stride in strides:
                layers.append(block(self.edge_planes, planes, dropout_rate, stride))
                self.edge_planes = planes
        else:
            for stride in strides:
                layers.append(block(self.in_planes, planes, dropout_rate, stride))
                self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        edge = self.canny(x)

        # edge_out = self.conv2(edge)
        out = self.conv4(torch.cat((x, edge), 1))

        # edge_out = self.layer1_edge(edge_out)
        out = self.layer1(out)
        # out = self.conv3(torch.cat((out, edge_out), 1))

        out = self.layer2(out)

        out = self.layer3(out)

        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    x = torch.randn(1, 3, 32, 32)
    net = Wide_ResNet_Edge(34, 10, 0.1, 10)
    net.to(device)
    output = net(x.to(device))
