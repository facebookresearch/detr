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
# from huffman import huffman_encode

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
    def forward(self, x):
        #**********************delta_zrl_m****************####
        In_max = x.max()
        In_min = x.min()
        c1 = part_quant(x, In_max, In_min, 8)
        x_256 = c1[0].clone()           #quant between 0-255
        [x0,x0_left] = delta_comp(x_256)
        mask_2bit = torch.where((x0 >= -1) & (x0 <= 2))
        mask_4bit = torch.where(((x0 < -1) & (x0 >= -7)) | ((x0 > 2) & (x0 <= 8)))
        mask_8bit = torch.where((x0 < -7) | (x0 > 8))
        #####################delta_zrl_m###################

        [x,x_left]= delta_comp(x)
        windowlen = self.windowlen
        # x = x 
        # In_max = torch.abs(x).max()
        en = 1
        # c = part_quant1(x, In_max, 16, en)
        # x1 = c
        In_max = x.max()
        In_min = x.min()
        c = part_quant(x, In_max, In_min, 16)
        x1 = c[0]*c[1] + c[2]           #quant between 0-255
        #######*************************bd verification***************###########
        # x1_origin = x1.clone()
        #######

        [batch,imgnum, height,width] = x1.shape
        y = 0
        y1 = 0
        y3 = 0
        b = x1.reshape(batch,imgnum,height,-1,windowlen).permute(0,1,3,2,4)                         #[1,32,28,112,4]
        c = b.reshape(batch,imgnum,int(width/windowlen),-1,windowlen,windowlen).permute(0,2,3,1,4,5) #[1,28,28,32,4,4]
        d = c.reshape(batch,int(width/windowlen),int(height/windowlen),imgnum,windowlen*windowlen).permute(0,1,2,4,3) #[1,28,28,16,32]
        d.squeeze_(0)
        d1 = d.permute(0,1,3,2) #wid len imgnums 16                             #[28,28,32,16]

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
        AT_A_inv = b_inv(AT_A)
        base = torch.matmul(AT_A_inv, a_base) #[2, 2, 3, 16]
        base_t = base.permute(0,1,3,2) #[2, 2, 16, 3]

        p = torch.matmul(base,d) #[2,2,3,3]
        r = torch.matmul(a_base_t,p) #[2,2,16,3]

        r1 = r.permute(0,1,3,2)

        # In_max = r1.max()
        # In_min = r1.min()
        # c1 = part_quant(r1, r1.max(), r1.min(),16)
        # r1 = c1[0]* c1[1] + c1[2]
        r1 = part_quant1(r1, In_max, 16, en)
        rr = torch.zeros(d1.shape)
        loss_fn = torch.nn.MSELoss(reduction='none')

        loss = loss_fn(d1,r1)
        loss_sum = loss.sum(dim = 3)
        loss_sum[mask] = self.loss + 1
        # print(loss_sum)
        mask1 = torch.where(loss_sum <= self.loss)

        #######*************************bd verification***************###########
        # x1_left   =  delta_comp_left(x1_origin)[0]
        # x1_right  =  delta_comp_right(x1_origin)[0]
        # x1_above  =  delta_comp_above(x1_origin)[0]
        # x1_below  =  delta_comp_below(x1_origin)[0]
        # r1_block  =  r1.clone()
        # r1_origin =  r1.reshape(int(width/windowlen),int(height/windowlen),imgnum,windowlen,windowlen) #[28,28,32,4,4]
        # r1_origin =  r1_origin.permute(2,1,3,0,4)                                                             #[32,28,4,28,4]
        # r1_origin =  r1_origin.reshape(imgnum,height,width).unsqueeze(0)                                      #[1,32,,112,112]
        # r1_left   =  delta_comp_left(r1_origin)[0]
        # r1_right  =  delta_comp_right(r1_origin)[0]
        # r1_above  =  delta_comp_above(r1_origin)[0]
        # r1_below  =  delta_comp_below(r1_origin)[0]
        # sub_left  =  torch.abs(r1_left - x1_left)
        # sub_right =  torch.abs(r1_right - x1_right)
        # sub_above =  torch.abs(r1_above - x1_above)
        # sub_below =  torch.abs(r1_below - x1_below)                                       #[1,32,112,112]

        # sub_left = sub_left.reshape(batch,imgnum,height,-1,windowlen).permute(0,1,3,2,4)                         #[1,32,28,112,4]
        # sub_left = sub_left.reshape(batch,imgnum,int(width/windowlen),-1,windowlen,windowlen).permute(0,2,3,1,4,5) #[1,28,28,32,4,4]
        # sub_left = sub_left.reshape(batch,int(width/windowlen),int(height/windowlen),imgnum,windowlen*windowlen).squeeze_(0) #[1,28,28,32,16]
        # sub_right = sub_right.reshape(batch,imgnum,height,-1,windowlen).permute(0,1,3,2,4)                         #[1,32,28,112,4]
        # sub_right = sub_right.reshape(batch,imgnum,int(width/windowlen),-1,windowlen,windowlen).permute(0,2,3,1,4,5) #[1,28,28,32,4,4]
        # sub_right = sub_right.reshape(batch,int(width/windowlen),int(height/windowlen),imgnum,windowlen*windowlen).squeeze_(0) #[1,28,28,32,16]
        # sub_above = sub_above.reshape(batch,imgnum,height,-1,windowlen).permute(0,1,3,2,4)                         #[1,32,28,112,4]
        # sub_above = sub_above.reshape(batch,imgnum,int(width/windowlen),-1,windowlen,windowlen).permute(0,2,3,1,4,5) #[1,28,28,32,4,4]
        # sub_above = sub_above.reshape(batch,int(width/windowlen),int(height/windowlen),imgnum,windowlen*windowlen).squeeze_(0) #[1,28,28,32,16]
        # sub_below = sub_below.reshape(batch,imgnum,height,-1,windowlen).permute(0,1,3,2,4)                         #[1,32,28,112,4]
        # sub_below = sub_below.reshape(batch,imgnum,int(width/windowlen),-1,windowlen,windowlen).permute(0,2,3,1,4,5) #[1,28,28,32,4,4]
        # sub_below = sub_below.reshape(batch,int(width/windowlen),int(height/windowlen),imgnum,windowlen*windowlen).squeeze_(0) #[1,28,28,32,16]

        # sub_bd = torch.zeros(sub_left.shape)
        # ##left
        # sub_bd[:,:,:,4] = sub_left[:,:,:,4]
        # sub_bd[:,:,:,8] = sub_left[:,:,:,8]
        # ##right
        # sub_bd[:,:,:,7] = sub_right[:,:,:,7]
        # sub_bd[:,:,:,11] = sub_right[:,:,:,11]
        # ##above
        # sub_bd[:,:,:,0:4] = sub_above[:,:,:,0:4]
        # ##below
        # sub_bd[:,:,:,12:16] = sub_below[:,:,:,12:16]

        # # sub_bd =  sub_bd.reshape(int(width/windowlen),int(height/windowlen),imgnum,windowlen,windowlen).permute(2,1,3,0,4) #[32,28,4,28,4]
        # # sub_bd =  sub_bd.reshape(imgnum,height,width)                                      #[32,,112,112]
        # # np.savetxt('sub_bd.txt',np.array(sub_bd[4].cpu()))

        # mask_1 = loss_sum <= 0 
        # mask_2 = loss_sum <= self.loss
        # mask_lossblock = torch.where(mask_2 > mask_1,True,False)
        # mask_lossblock_op = torch.where(mask_2 > mask_1,False,True)
        # bd_sub = sub_bd.sum(dim = 3)
        # bd_sub[mask_lossblock_op] = -1
        # median = bd_sub[mask_lossblock]
        # median, sortindex = torch.sort(median)
        # m_list = median.cpu().numpy().tolist()
        # # print("len=",len(m_list),round(len(m_list)/2)) 
        # med = median[round(len(m_list)/2)]
        # max_list = median[len(m_list)-1]

        # mask_3 = torch.where(bd_sub >= 2)
        # # print("mask_3_len=",len(mask_3[0]))
        # mask_4 = torch.where((bd_sub <= med) & (bd_sub >= 0))

        # # mask_th = torch.where(median >= 3)
        # # th = median[mask_th]
        # # th_list = th.cpu().numpy().tolist()
        # # print("med=",med)
        # # print("rate >3 :",len(th_list)/len(m_list))
        # # print("sub1",bd_sub[mask_3].shape)
        # # print("sub2",bd_sub[mask_4].shape)
        # # ##################
        
        
        
        rr = d1.clone()         #[28,28,32,16]
        rr[mask1] = r1[mask1]
        rr[:,:,:,1:0] = d1[:,:,:,1:0]
        rr[mask] = d1[mask]

        # #######*************************bd verification***************###########
        # rr[mask_3] = d1[mask_3]
        # rr[mask_4] = d1[mask_4]
        # mask3 = torch.where(mask_3[2]>1)
        # print("mask_3_len_2=",len(mask_3[0]))
        # ###################
        rr = rr.reshape(int(width/windowlen),int(height/windowlen),imgnum,windowlen,windowlen) #[28,28,32,4,4]
        rr = rr.permute(2,1,3,0,4)                                                             #[32,28,4,28,4]
        rr = rr.reshape(imgnum,height,width).unsqueeze(0)                                      #[1,32,,112,112]
        # y1 = len(mask[0])*imgnum
        y1 = len(mask[0])*imgnum
        mask2 = torch.where(mask1[2]<2)     #2 baseblock
        y3 = len(mask1[0]) - len(mask2[0])
        # #######*************************bd verification***************###########
        # y3 = len(mask1[0]) - len(mask2[0]) - len(mask_3[0])


        #**********************delta_zrl_m****************####
        mask_unable = torch.ones(d1.shape)
        mask_unable[mask] = 0
        mask_unable[mask1] = 0
        mask_unable[:,:,0:2,:] = 1
        # print ("unablelen=", len(mask_unable_where[0]))
        mask_unable = mask_unable.reshape(int(width/windowlen),int(height/windowlen),imgnum,windowlen,windowlen) #[28,28,32,4,4]
        mask_unable = mask_unable.permute(2,1,3,0,4)                                                             #[32,28,4,28,4]
        mask_unable = mask_unable.reshape(imgnum,height,width).unsqueeze(0)                                      #[1,32,,112,112]
        mask_unable_where = torch.where(mask_unable == 1)
        mask_unable_2bits = mask_unable[mask_2bit]
        mask_unable_4bits = mask_unable[mask_4bit]
        mask_unable_8bits = mask_unable[mask_8bit]
        mask_unable_2bits = torch.where(mask_unable_2bits == 1)
        mask_unable_4bits = torch.where(mask_unable_4bits == 1)
        mask_unable_8bits = torch.where(mask_unable_8bits == 1)
        # print ("unable_2bits_len=", len(mask_unable_2bits[0]))
        # print ("unable_4bits_len=", len(mask_unable_4bits[0]))
        # print ("unable_8bits_len=", len(mask_unable_8bits[0]))
        rate = (y1+3*y3+windowlen*windowlen*(imgnum*(int(width/windowlen)*int(height/windowlen))-y1-y3))/(windowlen*windowlen*(imgnum*(int(width/windowlen)*int(height/windowlen))))
        #2bits-8bits
        # rate = (8*y1+3*8*y3+windowlen*windowlen*imgnum*int(width/windowlen)*int(height/windowlen) + 2*len(mask_unable_2bits[0]) + 8*(len(mask_unable_8bits[0])+len(mask_unable_4bits[0])))/(8*windowlen*windowlen*(imgnum*(int(width/windowlen)*int(height/windowlen))))
        #4bits-8bits
        # rate = (8*y1+3*8*y3+windowlen*windowlen*imgnum*int(width/windowlen)*int(height/windowlen) + 4*(len(mask_unable_2bits[0])+len(mask_unable_4bits[0])) + 8*len(mask_unable_8bits[0]))/(8*windowlen*windowlen*(imgnum*(int(width/windowlen)*int(height/windowlen))))
        # print ("rate1=", rate1)

        # x_huff = x_256[mask_unable_where]
        # # print(x_huff,x_huff.shape)
        # huff_bits = huffman_encode(x_huff)
        # rate = (8*y1+3*8*y3+imgnum*int(width/windowlen)*int(height/windowlen) + huff_bits)/(8*windowlen*windowlen*(imgnum*(int(width/windowlen)*int(height/windowlen))))


        self.rate += rate
        self.temp += 1
        self.y1 += y1
        self.y3+= y3

        #---------------------------
        # f = open('result/layer{:0>2d}'.format(self.num)+'.txt','a')
        # f.write(str(self.rate)+'\n')
        # f.write(str(batch*imgnum*height*width)+' '+str(batch)+' '+str(imgnum)+' '+str(height)+' '+str(width)+'\n')
        if(self.temp %1 == 0):
            print('rate{}:  '.format(self.num),self.rate/self.temp)
            print('shape{}: '.format(self.num),batch,imgnum,height,width,batch*imgnum*height*width)
            # print('size{}:  '.format(self.num),batch*imgnum*height*width)
            print('ynum:  ',self.y1/self.temp,self.y3/self.temp,imgnum*(int(width/windowlen)*int(height/windowlen)))
        # f = open('loss_quan.txt', 'a')
        # loss_last = loss_fn(x1,rr)
        # f.write(str(loss_last.sum().item())+'\n')
        rr = delta_dcom(rr, x_left)
        return rr.cuda()

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
    # x_left[:,:,:,0] = x_c[:,:,:,0]
    # x_c[:,:,:,0] = 0
    return x_c, x_left

# #######*************************bd verification***************###########
def delta_comp_left(x):
    [batch,imgnum, height,width] = x.shape
    x_left = x[:,:,:,0:width-1]
    pad = torch.zeros(batch,imgnum,height,1).cuda()
    x_left = torch.cat((pad,x_left),dim=3)
    x_c = x-x_left
    x_left[:,:,:,0] = x_c[:,:,:,0]
    x_c[:,:,:,0] = 0
    return x_c, x_left

def delta_comp_right(x):
    [batch,imgnum, height,width] = x.shape
    x_left = x[:,:,:,1:width]
    pad = torch.zeros(batch,imgnum,height,1).cuda()
    x_left = torch.cat((x_left,pad),dim=3)
    x_c = x-x_left
    x_left[:,:,:,width-1] = x_c[:,:,:,width-1]
    x_c[:,:,:,width-1] = 0
    return x_c, x_left
def delta_comp_above(x):
    [batch,imgnum, height,width] = x.shape
    x_left = x[:,:,0:height-1,:]
    pad = torch.zeros(batch,imgnum,1,width).cuda()
    x_left = torch.cat((pad,x_left),dim=2)
    x_c = x-x_left
    x_left[:,:,0,:] = x_c[:,:,0,:]
    x_c[:,:,0,:] = 0
    return x_c, x_left
def delta_comp_below(x):
    [batch,imgnum, height,width] = x.shape
    x_left = x[:,:,1:height,:]
    pad = torch.zeros(batch,imgnum,1,width).cuda()
    x_left = torch.cat((x_left,pad),dim=2)
    x_c = x-x_left
    x_left[:,:,height-1,:] = x_c[:,:,height-1,:]
    x_c[:,:,height-1,:] = 0
    return x_c, x_left
########################
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


def part_quant(x, max, min, bitwidth):
    if max != min:
        Scale = (2 ** bitwidth - 1) / (max - min)
        Q_x = Round.apply((x - min) * Scale)
        return Q_x, 1 / Scale, min
    else:
        Q_x = x
        return Q_x, 1, 0

def part_quant1(x, max, bitwidth, en):
    if(en == 1):
        lsb = 2**(Round.apply(torch.log2(max/2**(bitwidth-1))) + 1)
    else:
        lsb = 2**(Round.apply(torch.log2(max/2**(bitwidth))) + 1)
    Q_x = Round.apply(x/lsb)*lsb
    return Q_x

class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        round = x.round()
        return round.to(x.device)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input, None, None

    

def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.solve(eye, b_mat)
    return b_inv