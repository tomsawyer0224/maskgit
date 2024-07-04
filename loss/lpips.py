import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from collections import namedtuple
#import os
#import hashlib
#import requests
#from tqdm import tqdm

class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.feature_net = VGG16()
        self.lins = nn.ModuleList([
            NetLinLayer(self.channels[0], use_dropout=True),
            NetLinLayer(self.channels[1], use_dropout=True),
            NetLinLayer(self.channels[2], use_dropout=True),
            NetLinLayer(self.channels[3], use_dropout=True),
            NetLinLayer(self.channels[4], use_dropout=True)
        ])#.load_state_dict(torch.load('./loss/lins_state_dict.pth'))
        self.load_lins_state_dict()
        for param in self.parameters():
            param.requires_grad = False
    def load_lins_state_dict(self):
        self.lins.load_state_dict(torch.load('./loss/lins_state_dict.pth'))

    def forward(self, real_x, fake_x):
        features_real = self.feature_net(self.scaling_layer(real_x))
        features_fake = self.feature_net(self.scaling_layer(fake_x))
        diffs = {}

        # calc MSE differences between features
        for i in range(len(self.channels)):
            diffs[i] = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2

        return sum([spatial_average(self.lins[i].model(diffs[i])) for i in range(len(self.channels))])

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale

class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout() if use_dropout else None,
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(weights=VGG16_Weights.DEFAULT).features
        slices = [vgg_pretrained_features[i] for i in range(30)]
        self.slice1 = nn.Sequential(*slices[0:4])
        self.slice2 = nn.Sequential(*slices[4:9])
        self.slice3 = nn.Sequential(*slices[9:16])
        self.slice4 = nn.Sequential(*slices[16:23])
        self.slice5 = nn.Sequential(*slices[23:30])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        vgg_outputs = namedtuple("VGGOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

def norm_tensor(x):
    """
    Normalize images by their length to make them unit vector?
    :param x: batch of images
    :return: normalized batch of images
    """
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + 1e-10)

def spatial_average(x):
    """
     imgs have: batch_size x channels x width x height --> average over width and height channel
    :param x: batch of images
    :return: averaged images along width and height
    """
    return x.mean([2,3], keepdim=True)
