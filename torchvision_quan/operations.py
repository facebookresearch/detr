
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn import functional as F
from compression_section import ex


class ReLuX(nn.Module):
    def __init__(self, args, mxRelu6=False,inplace=True):
        super(ReLuX, self).__init__()
        if mxRelu6:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        # print('-----------')
        input = self.relu(input)
        return input

# class ConvX(nn.Conv2d):
#     count = 0
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(ConvX, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                         padding, dilation, groups, bias)
#         windowlen = 4
#         lossthred = 0
#         self.bit = 16
#         self.count = ConvX.count

#         self.ex = ex(self.count,windowlen,lossthred)

#         ConvX.count+= 1
#     #     return input
#     def forward(self, x):

#         In_min = x.min()
#         In_max = x.max()
#         c = part_quant(x, In_max, In_min, self.bit)
#         x = c[0]*c[1]+c[2]

#         x = self.ex(x)
#         # In_mean = x.mean()
#         # x = x - In_mean
#         # In_max = torch.abs(x).max()
#         # c = part_quant1(x, In_max, self.bit)
#         # x = c+In_mean

#         x = F.conv2d(x, self.weight, self.bias, self.stride,
#                             self.padding, self.dilation, self.groups)
        
#         return x

class ConvX(nn.Conv2d):
    count = 0
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ConvX, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        self.bit = 16
        self.windowlen = 4

        self.count = ConvX.count
        ConvX.count += 1
        # f = open('quan.txt')
        # self.lines = f.readlines()
        # self.len = len(self.lines)
    #     return input
    def forward(self, x):
        # ex_=ex(0,4,0)
        # x=ex_(x)
        # if(self.count < self.len):
        #     ex_ = ex(self.count,4,float(self.lines[self.count].split(' ')[0]))
        # else:
        #     ex_ = quan(16)
        # # print(input.shape)
        # if(x.shape[1] > 3):
        #     if(x.shape[2] > 4):
        #         x = ex_(x)
        #     else:
        #         exq = quan(16)
        #         x = exq(x)
        # else:
        #     x = ex_(x)
        #------------------------------------------

        x = F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        return x



def part_quant(x, max, min, bitwidth):
    if max != min:
        Scale = (2 ** bitwidth - 2) / (max - min)
        Q_x = Round.apply((x - min) * Scale)
        return Q_x, 1 / Scale, min
    else:
        Q_x = x
        return Q_x, 1, 0


class quan(nn.Module):
    def __init__(self,bit):
        super().__init__()
        self.bitwidth = bit
    def forward(self, x):
        max = x.max()
        min = x.min()
        # Scale = (2 ** self.bitwidth - 2) / (max - min)
        # Q_x = Round.apply((x - min) * Scale)
        # c = Q_x *(1 / Scale) + min
        c = part_quant(x, max, min, self.bitwidth)
        x = c[0]*c[1]+c[2]
        return x


def part_quant1(x, max, bitwidth, en):
    if(en == 1):
        lsb = 2**(Round.apply(torch.log2(max/2**(bitwidth-1))) + 1)
    else:
        lsb = 2**(Round.apply(torch.log2(max/2**(bitwidth))) + 1)
    Q_x = Round.apply(x/lsb)*lsb
    return Q_x
    # return x

class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        round = x.round()
        return round.to(x.device)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input, None, None
