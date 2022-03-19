#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchsummary import summary
import torchvision.transforms.functional as tvF


from unet import UNet
from utils import *

import os
import json
from argparse import ArgumentParser
from torchvision import models, transforms

# load the model

def parse_args():
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018) into rain2rain')

    parser.add_argument('-i', '--image', required=True,
        help='path to image')
    parser.add_argument('-c', '--load-ckpt',
        help='load model checkpoint')

    return parser.parse_args()
args = parse_args()

ckpt_fname = args.load_ckpt
model = UNet(in_channels=3)
model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
# get all the model children as list
model_children = list(model.children())
print("cut")
print(model_children)
print("cut")

counter = 0 
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential: #i is the actual block instead of layer
        for child in model_children[i]:
            print('a')
            if type(child) == nn.Conv2d:
                counter += 1
                model_weights.append(child.weight)
                conv_layers.append(child)
    print(i)
    print(model_children[i])
print(f"Total convolutional layers: {counter}")



# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")




plt.figure(figsize=(20, 17))
for i, filter in enumerate(model_weights[0]):
    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig('../filter.jpg')
plt.show()

# read and visualize an image
#img = cv.imread(f"{args.image}")
#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
imgs = os.listdir(args.image)
img_path = os.path.join(args.image, imgs[0])
img =  Image.open(img_path).convert('RGB')
plt.imshow(img)
plt.show()
# define the transforms
#transform = transforms.Compose([
 #   transforms.ToPILImage(),
  #  transforms.Resize((512, 512)),
   # transforms.ToTensor(),
#])
#img = np.array(img)
img=tvF.to_tensor(img)
# apply the transforms
#img = transform(img)
print(img.size())
# unsqueeze to add a batch dimension
img = img.unsqueeze(0)
print(img.size())

# pass the image through all the layers
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results

for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"..layer_{num_layer}.jpg")
    # plt.show()
    plt.close()