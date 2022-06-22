import math
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn





class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)


#真正起作用的量化函数
class Quantizer(Function):
    @staticmethod
    def forward(ctx, input, nbit):
        scale = 2 ** nbit - 1
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize(input, nbit):
    return Quantizer.apply(input, nbit)

 
def dorefa_w(w, nbit_w):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        #将权重限制在[0,1]之间
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        #权重量化
        w = 2 * quantize(w, nbit_w) - 1

    return w


def dorefa_a(input, nbit_a):
    return quantize(torch.clamp(0.1 * input, 0, 1), nbit_a)

def getradix(xmax, bit_width):
    if xmax == 0:
        return 0
    # elif np.isnan(xmax.detach()):
    #     return 0
    radix = bit_width - 1 - (math.floor(math.log2(xmax) + 1))
    return radix

def quan_w(w, nbit_w):
    xmax = torch.max(torch.abs(w))
    radix = getradix(xmax,nbit_w)
    scale = 2**radix
    w = torch.round(w * scale) / scale
    return w

def quan_a(x, nbit_a):
    xmax = torch.max(torch.abs(x))
    radix = getradix(xmax,nbit_a)
    scale = 2**radix-1
    w = torch.round(x * scale) / scale
    return x

class QuanConv(nn.Conv2d):
    """docstring for QuanConv"""
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=32,
                 nbit_a=32, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False):
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        # name_w_dict = {'dorefa': dorefa_w}
        name_w_dict = {'dorefa': quan_w}
        # name_a_dict = {'dorefa': dorefa_a}
        name_a_dict = {'dorefa': quan_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]

    # @weak_script_method
    def forward(self, input):
        
        if self.nbit_w <=32:
            #量化卷积
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            #卷积保持不变
            w = self.weight

        # if self.nbit_a <=32:
        #     #量化激活
        #     x = self.quan_a(input, self.nbit_a)
        # else:
        #     #激活保持不变
        #     x = input
        # print('x unique',np.unique(x.detach().numpy()).shape)
        # print('w unique',np.unique(w.detach().numpy()).shape)

        #做真正的卷积运算
        x = input
        
        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output


class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=32, nbit_a=32):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight

        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a)
        else:
            x = input

        # print('x unique',np.unique(x.detach().numpy()))
        # print('w unique',np.unique(w.detach().numpy()))

        output = F.linear(x, w, self.bias)

        return output


