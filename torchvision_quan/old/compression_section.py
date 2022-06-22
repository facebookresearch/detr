import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from scipy.linalg import block_diag
from numpy import average, linalg, dot, mat, array 
from sklearn.linear_model import LinearRegression
 
class expressimg(nn.Module):
    def __init__(self,windowlen,losss, num):
        super().__init__()
        self.windowlen = windowlen
        self.loss = losss
        self.num = num
        self.temp = 0
        self.rate = 0
        self.y1 = 0
        self.y3 = 0
        self.baseflag = 0
        self.bit = 8
    def forward(self, x):
        [x,x_left]= delta_comp(x)
        windowlen = self.windowlen
        x = x 
        In_max = torch.abs(x).max()
        en = 1
        c = part_quant1(x, In_max, self.bit, en)
        x1 = c


        [batch,imgnum, height,width] = x1.shape
        y = 0
        y1 = 0
        y3 = 0
        b = x1.reshape(batch,imgnum,height,-1,windowlen).permute(0,1,3,2,4)
        c = b.reshape(batch,imgnum,int(width/windowlen),-1,windowlen,windowlen).permute(0,2,3,1,4,5)
        d = c.reshape(batch,int(width/windowlen),int(height/windowlen),imgnum,windowlen*windowlen).permute(0,1,2,4,3)
        d.squeeze_(0)
        d1 = d.permute(0,1,3,2) #wid len imgnums 16

        a_base1 = d1[:,:,0,:].reshape(d1.shape[0],d1.shape[1],1,d1.shape[3])
        a_base2 = d1[:,:,1,:].reshape(d1.shape[0],d1.shape[1],1,d1.shape[3])
        a_base3 = torch.ones(a_base1.shape).cuda()
        a_base = torch.cat((a_base1,a_base2,a_base3),2)

        max1 = a_base1.max(dim = 3)[0].squeeze(2)
        min1 = a_base1.min(dim = 3)[0].squeeze(2)
        max2 = a_base2.max(dim = 3)[0].squeeze(2)
        min2 = a_base2.min(dim = 3)[0].squeeze(2)   
        mask = torch.where((max1 - min1 < 1e-6)&(max2 - min2 < 1e-6))

        a_base_t = a_base.permute(0,1,3,2) #[2, 2, 16, 3]
        AT_A = torch.matmul(a_base,a_base_t)
        eye1 = torch.eye(3).cuda()
        maskk = torch.where(torch.det(AT_A)==0)
        AT_A[maskk] = eye1
        AT_A_inv = torch.inverse(AT_A)
        base = torch.matmul(AT_A_inv, a_base) #[2, 2, 3, 16]
        base_t = base.permute(0,1,3,2) #[2, 2, 16, 3]

        p = torch.matmul(base,d) #[2,2,3,3]
        r = torch.matmul(a_base_t,p) #[2,2,16,3]

        r1 = r.permute(0,1,3,2)

        r1 = part_quant1(r1, In_max, self.bit, en)

        rr = torch.zeros(d1.shape)
        loss_fn = torch.nn.MSELoss(reduction='none')

        loss = loss_fn(d1,r1)
        loss_sum = loss.sum(dim = 3)
        mask1 = torch.where(loss_sum <= self.loss)
        rr = d1
        rr[mask1] = r1[mask1]
        rr[:,:,:,1:0] = d1[:,:,:,1:0]
        rr[mask] = d1[mask]
        rr = rr.reshape(int(width/windowlen),int(height/windowlen),imgnum,windowlen,windowlen)
        rr = rr.permute(2,1,3,0,4)
        rr = rr.reshape(imgnum,height,width).unsqueeze(0)
        y1 = len(mask[0])*imgnum
        mask2 = torch.where(mask1[2]<2)
        y3 = len(mask1[0]) - len(mask2[0])
        rate = (y1+3*y3+windowlen*windowlen*(imgnum*(int(width/windowlen)*int(height/windowlen))-y1-y3))/(windowlen*windowlen*(imgnum*(int(width/windowlen)*int(height/windowlen))))
        self.rate += rate
        self.temp+=1
        self.y1 += y1
        self.y3+= y3
        #-------------------------------
        f = open('result.txt','a')
        f.write(str(self.rate)+'\n')
        f.write(str(batch*imgnum*height*width)+'\n')
        if(self.temp %1 == 0):
            print('rate{}:  '.format(self.num),self.rate/self.temp)
            print('shape{}: '.format(self.num),batch,imgnum,height,width,batch*imgnum*height*width)
            print('size{}:  '.format(self.num),batch*imgnum*height*width)
            print('ynum:  ',self.y1/self.temp,self.y3/self.temp,imgnum*(int(width/windowlen)*int(height/windowlen)))
        # f = open('loss_quan.txt', 'a')
        # loss_last = loss_fn(x1,rr)
        # f.write(str(loss_last.sum().item())+'\n')
        rr = delta_dcom(rr, x_left)
        return rr

class ex(nn.Module):
    def __init__(self,num=0,windowlen=4,loss=0):
        super().__init__()
        self.windowlen = windowlen
        self.loss = loss
        self.num = num
        self.encoder = expressimg(self.windowlen,self.loss,num)
    def forward(self, x):
        [layer,imgnums, height,width] = x.shape
        
        modw = width % self.windowlen
        if modw > 0:
            padding = self.windowlen - modw
            left = int(padding/2)
            right = int((padding+1)/2)
            width1 = width+padding
            paddingw = (left, right, 0, 0)
            padw = nn.ZeroPad2d(paddingw)
            x = padw(x)
        else:
            width1= width
                
        modh = height % self.windowlen
        if modh > 0:
            padding = self.windowlen - modh
            left = int(padding/2)
            right = int((padding+1)/2)
            height1 = height+padding
            paddingh = (0, 0, left, right)
            padh = nn.ZeroPad2d(paddingh)
            x = padh(x)
        else:
            height1 = height
        enclist = self.encoder(x)
        if modw > 0:  
            enclist = enclist[:, :, :, paddingw[0]:paddingw[0]+width]
        if modh > 0:
            enclist = enclist[:,:, paddingh[2]:paddingh[2]+height, :]
        return enclist

def delta_comp(x):
    [batch,imgnum, height,width] = x.shape
    x_left = x[:,:,:,0:width-1]
    pad = torch.zeros(batch,imgnum,height,1).cuda()
    x_left = torch.cat((pad,x_left),dim=3)
    x_c = x-x_left
    return x_c, x_left

def delta_dcom(x_c, x_left):
    x_d = x_c+x_left
    return x_d

def save_image0(img_num,img_xx,im,len,idx,num,addrx,addry,win):
    # print(im.size)
    rate1 = float(im.size[0]/len)
    rate2 = float(im.size[1]/len)
    win1 = win * rate1
    win2 = win * rate2
    # print(win1,win2)
    addrx = addrx
    addry = addry
    if im.mode != 'RGB':
        im = im.convert('RGB')
    draw = ImageDraw.Draw(im)
    draw.rectangle([addrx*win1,addry*win2,addrx*win1+win1,addry*win2+win2], width= 2, fill ="yellow", outline ="pink") 
    Image.fromarray(np.asarray(im)).save('img{}/result{}/result_origin{}.jpg'.format(img_num,idx,num))

    draw = ImageDraw.Draw(img_xx)
    draw.rectangle([addrx*win,addry*win,addrx*win+win,addry*win+win], width= 1, fill ="yellow", outline ="pink") 
    Image.fromarray(np.asarray(img_xx)).save('img{}/result{}/result_gray{}.jpg'.format(img_num,idx,num))
    return img_xx,im

def save_image1(img_loss,img_num,im,idx,num,addrx,addry,win):
    addrx = addrx
    addry = addry
    if im.mode != 'RGB':
        im = im.convert('RGB')
    draw = ImageDraw.Draw(im)
    # draw.rectangle([addrx*win,addry*win,addrx*win+win,addry*win+win], width= 1, outline ="pink") 
    draw.rectangle([addrx*win,addry*win,addrx*win+win,addry*win+win], width= 1, outline ="pink")
    draw.text((addrx*win,addry*win), str(format(img_loss,'.5g')), fill = (0, 0 ,0))
    Image.fromarray(np.asarray(im)).save('img{}/result{}/result_gray_loss{}.jpg'.format(img_num,idx,num))
    return im

def save_image2(img_value1,img_value2,img_num,im,im1,idx,num,addrx,addry,win):
    addrx = addrx
    addry = addry
    if im1.mode != 'RGB':
        im1 = im1.convert('RGB')
    draw = ImageDraw.Draw(im1)
    # draw.rectangle([addrx*win,addry*win,addrx*win+win,addry*win+win], width= 1, outline ="pink") 
    draw.rectangle([addrx*win,addry*win,addrx*win+win,addry*win+win], width= 1, outline ="pink")
    draw.text((addrx*win,addry*win), str(format(img_value1,'.5g')), fill = (256, 0 ,0))
    draw.text((addrx*win,addry*win+20), str(format(img_value2,'.5g')), fill = (256, 0 ,0))
    Image.fromarray(np.asarray(im1)).save('img{}/result{}/result_gray_value-{}.jpg'.format(img_num,idx,num))


    if im.mode != 'RGB':
        im = im.convert('RGB')
    draw = ImageDraw.Draw(im)
    # draw.rectangle([addrx*win,addry*win,addrx*win+win,addry*win+win], width= 1, outline ="pink") 
    draw.rectangle([addrx*win,addry*win,addrx*win+win,addry*win+win], width= 1, outline ="pink")
    draw.text((addrx*win,addry*win), str(format(img_value1,'.5g')), fill = (256, 0 ,0))
    draw.text((addrx*win,addry*win+20), str(format(img_value2,'.5g')), fill = (256, 0 ,0))
    Image.fromarray(np.asarray(im)).save('img{}/result{}/result_gray_value1-{}.jpg'.format(img_num,idx,num))
    return im,im1


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
