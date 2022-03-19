#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable

from irnn import irnn
from backbone.resnext.resnext101_regular import ResNeXt101



def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class Spacial_IRNN(nn.Module):
    def __init__(self,in_channels,alpha=1.0):
        super(Spacial_IRNN,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.left_weight.weight  = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))    
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.up_weight.weight    = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.down_weight.weight  = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))

    def forward(self,input):
        return irnn.apply(input,self.up_weight.weight,self.right_weight.weight,self.down_weight.weight,self.left_weight.weight, self.up_weight.bias,self.right_weight.bias,self.down_weight.bias,self.left_weight.bias)

class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out

class DSC_Module(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1,alpha=1.0):
        super(DSC_Module,self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels,alpha)
        self.irnn2 = Spacial_IRNN(self.out_channels,alpha)
        self.conv_in = conv1x1(in_channels,in_channels)
        self.conv2 = conv1x1(in_channels*4,in_channels)
        self.conv3 = conv1x1(in_channels*4,in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        
        
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv2(out)
        top_up,top_right,top_down,top_left = self.irnn2(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        
        return out

class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class Predict(nn.Module):
    def __init__(self, in_planes=32, out_planes=1, kernel_size=1):
        super(Predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size)

    def forward(self, x):
        y = self.conv(x)

        return y


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3):
        super(UNet, self).__init__()
        
        self.encoder1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.bne1 = nn.InstanceNorm2d(32)
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.bne2 = nn.InstanceNorm2d(64)
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bne3 = nn.InstanceNorm2d(128)
        self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.encoder6=   nn.Conv2d(512, 1024, 3, stride=1, padding=1)

        self.dsc1 = DSC_Module(32, 32,alpha=0.8)#edited
        self.dsc2 = DSC_Module(32, 32,alpha=0.8)#edited
        #self.dsc3 = DSC_Module(64, 64,alpha=0.8)#edited

        self.decoder1=   nn.Conv2d(1024,512, 3, stride=1, padding=1)
        self.decoder2 =   nn.Conv2d(512, 256, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.bnd1 = nn.InstanceNorm2d(64)
        self.decoder3 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.bnd2 = nn.InstanceNorm2d(32)
        self.decoder4 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bnd3 = nn.InstanceNorm2d(16)
        self.decoder5 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder6 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv2d(512, 1024, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bndf1 = nn.InstanceNorm2d(64)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bndf2 = nn.InstanceNorm2d(32)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.bndf3 = nn.InstanceNorm2d(16)
        self.decoderf4 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.decoderf5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bnef1 = nn.InstanceNorm2d(32)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bnef2 = nn.InstanceNorm2d(64)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bnef3 = nn.InstanceNorm2d(128)
        self.encoderf4 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.encoderf5 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.final = nn.Conv2d(16,3,1,stride=1,padding=0)
        self.bnf = nn.InstanceNorm2d(3)

        self.tmp1 = nn.Conv2d(64,32,1,stride=1,padding=0)
        self.bnt1 = nn.InstanceNorm2d(32)
        self.tmp2 = nn.Conv2d(128,32,1,stride=1,padding=0)
        # self.bnt2 = nn.BatchNorm2d(32)
        self.tmp3 = nn.Conv2d(64,32,1,stride=1,padding=0)

        self.tmpf3 = nn.Conv2d(128,32,1,stride=1,padding=0)
        self.tmpf2 = nn.Conv2d(64,32,1,stride=1,padding=0)
        self.tan = nn.Tanh()
        
        # self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):
        #reverse unet start

        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear')) #edited
        t1 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        
        o1 = out1
        o1 = self.dsc2(o1) #edited

        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        
        t2 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')
        o2 = out1        
        

        # U-NET encoder start
        

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2)) #edited
        #Fusing all feature maps from K-NET
        out = torch.add(out,torch.add(t1,self.tmp1(t2))) #edited
        
        u1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        u2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        u3=out
        out = F.relu(F.max_pool2d(self.encoder4(out),2,2))
        u4=out
        out = F.relu(F.max_pool2d(self.encoder5(out),2,2))
        u5 = out
        out = F.relu(F.max_pool2d(self.encoder6(out),2,2))
        
        u1 = self.dsc1(u1) #edited

        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u5)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u4)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u3)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u2)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u1)
        # out = F.relu(F.interpolate(self.decoder6(out),scale_factor=(2,2),mode ='bilinear'))
        

        #reverse unet resume
        out1 = torch.add(out1,o2)
        
        t2 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')

        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out1 = torch.add(out1,o1)
        t1 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))
        
        # Fusing all layers at the last layer of decoder
        # print(out.shape,t1.shape,t2.shape,t3.shape)
        out = torch.add(out,torch.add(t1,self.tmpf2(t2)))#edited

        out = F.relu(F.interpolate(self.decoder6(out),scale_factor=(2,2),mode ='bilinear'))
        
        out = F.relu(self.final(out))

        return self.tan(out)
