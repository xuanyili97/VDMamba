## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init

from utils import *
from networks.ConvLSTM import ConvLSTM
from networks.SoftMedian import softMedian, softMin, softMax
from networks.LiteFlowNet import backwarp
import torch.nn.functional as F

class ResBlock3d(nn.Module):
    def __init__(self, din, n_block=1):
        super().__init__()
        self.n_block = n_block
        self.res_convs = nn.ModuleList()
        self.prelus = nn.ModuleList()
        for i in range(self.n_block):
            self.res_convs.append(
                nn.Sequential(
                    nn.Conv3d(din, din, (3, 3, 3), 1, (1,1,1)),
                    nn.PReLU(),
                    nn.Conv3d(din, din, (3, 3, 3), 1, (1,1,1)),
                    nn.PReLU()
                )
            )
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        for i in range(self.n_block):
            resx = x
            x = self.prelus[i](self.res_convs[i](x)) + resx
        return x

class ResBlock2d(nn.Module):
    def __init__(self, din, n_block=1):
        super().__init__()
        self.n_block = n_block
        self.res_convs = nn.ModuleList()
        self.prelus = nn.ModuleList()
        for i in range(self.n_block):
            self.res_convs.append(
                nn.Sequential(
                    nn.Conv2d(din, din, (3, 3), 1, (1,1)),
                    nn.PReLU(),
                    nn.Conv2d(din, din, (3, 3), 1, (1,1)),
                    nn.PReLU()
                )
            )
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        for i in range(self.n_block):
            resx = x
            x = self.prelus[i](self.res_convs[i](x)) + resx
        return x

class Block(nn.Module):
    def __init__(self, nc_out=3, nc_ch=32):
        super(Block, self).__init__()
        
        self.epoch = 0

        self.conv3d_in = nn.Conv3d(3, nc_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock3d = ResBlock3d(nc_ch, 2)
        self.conv3d_out = nn.Conv3d(nc_ch, nc_out, kernel_size=3, stride=1, padding=1)

        self.conv2d_in = nn.Conv2d(3*7, nc_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock2d = ResBlock2d(nc_ch, 2)
        self.conv2d_out = nn.Conv2d(nc_ch, 22+12, kernel_size=3, stride=1, padding=1)

        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        F1 = self.relu(self.conv3d_in(X))
        F2 = self.ResBlock3d(F1)

        F3 = self.conv3d_out(F2)
        b, c, sq, h, w = F3.shape
        F3 = F3.view(b, c*sq, h, w)
        F4 = self.conv2d_in(F3)
        F5 = self.ResBlock2d(F4)
        F6 = self.conv2d_out(F5)
        Y, mask, flow = F6[:,:21], F6[:,21:22], F6[:,22:]
        mask = self.sigmoid(mask)
        Y = Y.view(b, c, sq, h, w)
        Y = softMedian(Y, 2, )

        # softmedian = torch.median(X, 2, keepdims=True)[0]
        # print(softmedian.shape, Y.shape)
        return mask * Y + (1 - mask) * X[:,:,3], flow

class Model(nn.Module):
    def __init__(self, nc_out=3, nc_ch=32, device='cuda'):
        super(Model, self).__init__()

        self.epoch = 0

        self.block1 = Block(3, 32)
        self.block2 = Block(3, 32)
        self.block3 = Block(3, 32)
        self.blocks = [self.block1, self.block2, self.block3]

    def batch_warp(self, x, flow):
        # flow: b, sq_*2, h, w
        print(flow.shape, x.shape)
        b, c, sq_, h, w = x.shape
        x_ = x.transpose(1,2).contiguous().view(-1, c, h, w)
        flow_ = flow.view(b, sq_, 2, h, w).view(-1, 2, h, w)
        return backwarp(x_, flow_)

    def forward(self, X, scales=[4,2,1]):
        b, c, sq, h, w = X.shape
        flow_list = []
        x_prev = None
        for i, scale in enumerate(scales):
            size = [sq, h//scale, w//scale]
            x_scale = F.interpolate(X, size=size, mode="trilinear", align_corners=False)
            if x_prev is not None:
                x_prev = F.interpolate(x_prev, size=size[-2:], mode="bilinear", align_corners=False)
                x_scale[:,:,sq//2] += x_prev
            x_prev, flow = self.blocks[i](x_scale)
            flow_list.append(flow)
        return x_prev, flow_list
