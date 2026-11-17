import torch.nn as nn
import torch

class dispHead(nn.Module):
    def __init__(self):
        super(dispHead, self).__init__()

        self.convd = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1,padding=0, bias=True))

    def forward(self, x):
        return self.convd(x)


class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = self.convf2 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=0),
            torch.nn.LeakyReLU(inplace=True))

        self.convf1 = torch.nn.Sequential(
            nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1,padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True))
        self.convf2 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True))

        self.convo1 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=160, out_channels=127, kernel_size=3, stride=1,padding=0),
            torch.nn.LeakyReLU(inplace=True))
        self.convo2 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=0),
            torch.nn.LeakyReLU(inplace=True))

    def forward(self, depth, corr):
        cor = self.convc1(corr)
 
        dep = self.convf1(depth)
        dep = self.convf2(dep)

        out = torch.cat([cor, dep], dim=1)
        out = self.convo1(out)
        out = torch.cat([out, depth], dim=1)
        out = self.convo2(out)
        return out


class BasicUpdateBlock(nn.Module):
    def __init__(self):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()
        self.flow_head = dispHead()
        self.mask = nn.Conv2d(256, 64 * 9, 1,padding=0)

    def forward(self, net, corr, depth):
        net = self.encoder(depth, corr)
        delta_depth = self.flow_head(net)
        mask = .25 * self.mask(net)

        return net, mask, delta_depth

