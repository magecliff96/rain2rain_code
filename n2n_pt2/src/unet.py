#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import torchvision.models as models
from torch.autograd import Variable

from deform_conv_v2 import * 
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

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()
        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        self._block4m = nn.Sequential(
            nn.Conv2d(240, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))264+144 = 408

        self._block5m = nn.Sequential(
            nn.Conv2d(288, 144, 3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()
        

        #Directional Dilated Convolution
        self.ddc1= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)


        weights2 = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).unsqueeze(0)
        weights2.requires_grad = True
        self.ddc2= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            self.ddc2.weight = nn.Parameter(weights2*self.ddc2.weight)

        weights3 = torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).unsqueeze(0)
        weights3.requires_grad = True
        self.ddc3= nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            self.ddc3.weight = nn.Parameter(weights2*self.ddc2.weight)



        ##IRNN
        #self.dsc = DSC_Module(48, 48,alpha=0.8)#edited

        ##deformable
        #self.defc=DeformableConv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)

        ##utilities
        self.bn48=nn.BatchNorm2d(48)


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        ##dsc
        #mege21=F.relu(self.bn48(self.dsc(pool2)))
        #mege22=F.relu(self.bn48(self.dsc(pool2)))
        #mege23=F.relu(self.bn48(self.dsc(pool2)))
        #mege41=F.relu(self.bn48(self.dsc(pool4)))
        #mege42=F.relu(self.bn48(self.dsc(pool4)))
        #mege43=F.relu(self.bn48(self.dsc(pool4)))
        
        
        ##defc
        #mege21=F.relu(self.bn48(self.defc(pool2)))
        #mege22=F.relu(self.bn48(self.defc(pool2)))
        #mege23=F.relu(self.bn48(self.defc(pool2)))
        #mege41=F.relu(self.bn48(self.defc(pool4)))
        #mege42=F.relu(self.bn48(self.defc(pool4)))
        #mege43=F.relu(self.bn48(self.defc(pool4)))

        #ddc
        mege21=F.relu(self.bn48(self.ddc1(pool2)))
        mege22=F.relu(self.bn48(self.ddc2(pool2)))
        mege23=F.relu(self.bn48(self.ddc3(pool2)))

        mege41=F.relu(self.bn48(self.ddc1(pool4)))
        mege42=F.relu(self.bn48(self.ddc2(pool4)))
        mege43=F.relu(self.bn48(self.ddc3(pool4)))

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4, mege41, mege42, mege43), dim=1)#
        upsample4 = self._block4m(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2, mege21, mege22, mege23), dim=1)#
        upsample2 = self._block5m(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)

        concat1 = torch.cat((upsample1, x), dim=1)
        final = self._block6(concat1) + x #18 conv
        # Final activation
        return [final, mege21]
