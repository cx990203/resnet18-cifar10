import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in: 输入通道
        :param ch_out: 输出通道
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        res = F.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        res = res + self.extra(x)
        return res


class Resnet18(nn.Module):

    def __init__(self):
        super(Resnet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
        )
        self.blk1 = ResBlk(64, 128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Sequential(
            nn.Linear(512 * 1 * 1, 10)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


if __name__ == '__main__':
    data = torch.randn([2, 3, 32, 32])
    net = Resnet18()
    out = net(data)
    print(out.size())
