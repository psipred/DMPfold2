import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from math import sqrt

NUM_CHANNELS = 442

class Maxout2d(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, kernel_size=1, dilation=1, block=0):
        super(Maxout2d, self).__init__()
        self.in_channels, self.out_channels, self.pool_size = in_channels, out_channels, pool_size
        self.lin = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * pool_size, kernel_size=kernel_size, dilation=dilation, padding=dilation*(kernel_size-1)//2)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        if block > 0:
            nn.init.xavier_uniform_(self.lin.weight, gain=1.0 / sqrt(block))
        else:
            nn.init.xavier_uniform_(self.lin.weight, gain=1.0)

    def forward(self, inputs):
        x = self.lin(inputs)

        N, C, H, W = x.size()

        x = x.view(N, C//self.pool_size, self.pool_size, H, W)
        x = x.max(dim=2)[0]
        x = self.norm(x)

        return x


class CSE(nn.Module):
    def __init__(self, width, reduction=16):
        super(CSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(width, width // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(width // reduction, width, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, _, _ = x.size()
        y = self.avg_pool(x).view(N, C)
        y = self.fc(y).view(N, C, 1, 1)

        return x * y.expand_as(x)


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):

        y = self.conv(x)
        y = torch.sigmoid(y)

        return x * y


class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        return cSE + sSE


# ResNet Module
class ResNet_Block(nn.Module):
    def __init__(self,width,fsize,dilv,nblock):
        super(ResNet_Block, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.layer1 = Maxout2d(in_channels=width, out_channels=width, pool_size=4, kernel_size=fsize, dilation=dilv, block=nblock)
        self.scSE = SCSE(width, 16)

    def forward(self, x):

        residual = x
        out = self.dropout1(x)
        out = self.dropout2(out)
        out = self.layer1(out)
        out = self.scSE(out)
        out = out + residual

        return out


# RNNResNet Module
class GRUResNet(nn.Module):
    def __init__(self,width,cwidth):
        super(GRUResNet, self).__init__()

        self.width = width
        self.cwidth = cwidth

        self.embed = nn.Embedding.from_pretrained(torch.eye(22), freeze=True)
        self.vgru = nn.GRU(22, width, batch_first=False, num_layers=2, bidirectional=False)
        self.hgru = nn.GRU(width, width//2, batch_first=False, num_layers=2, dropout=0.1, bidirectional=True)

        layers = []

        layer = Maxout2d(in_channels=NUM_CHANNELS+width, out_channels=cwidth, pool_size=3)

        layers.append(layer)

        nblock = 1

        for rep in range(16):
            for fsize,dilv in [(5,1)]:
                if fsize > 0:
                    layer = ResNet_Block(cwidth, fsize, dilv, nblock)
                    layers.append(layer)
                    nblock += 1
                    
        layer = nn.Conv2d(in_channels=cwidth, out_channels=2+3*34, kernel_size=1)
        layer.weight.data.zero_()
        layer.bias.data.zero_()
        layers.append(layer)

        self.resnet = nn.Sequential(*layers)

    def forward(self, x, x2):

        nseqs = x.size()[0]
        nres = x.size()[1]

        x = self.embed(x)
        xx = torch.empty((0, self.width), device=x.device)
        for i in range(0, nres, 50):
            xx = torch.cat((xx, self.vgru(x[:,i:min(nres, i+50)])[0][-1,:,:]))
        x = self.hgru(xx.unsqueeze(1))[0]

        mat1d = x.permute(1,2,0)
        mat2d = mat1d.unsqueeze(2).expand(1, self.width, nres, nres)
        mat2d_T = mat1d.unsqueeze(3).expand(1, self.width, nres, nres)
        x = mat2d * mat2d_T

        x = torch.cat((x, x2), dim=1)

        #if x.requires_grad:
            # now call the checkpoint API and get the output
            #out = checkpoint_sequential(self.resnet, 4, x)
        #else:
        #print(x)
        out = self.resnet(x)

        return out
