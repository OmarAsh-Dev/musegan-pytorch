import torch
from torch import nn
from .utils import Reshape


class BarGenerator(nn.Module):
    
    n_steps_per_bar = 16
    n_pitches = 84
    
    def __init__(self,
                 z_dimension: int=32,
                 hid_features: int=1024,
                 hid_channels: int=512,
                 out_channels: int=1):
        super().__init__()
        self.net = nn.Sequential(
            # input shape: (batch_size, 4*z_dimension)
            nn.Linear(4 * z_dimension, hid_features),
            nn.BatchNorm1d(hid_features),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_features)
            Reshape(shape=[hid_channels, hid_features//hid_channels, 1]),
            # output shape: (batch_size, hid_channels, hid_features//hid_channels, 1)
            nn.ConvTranspose2d(hid_channels, hid_channels,
                               kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels, 2*hid_features//hid_channels, 1)
            nn.ConvTranspose2d(hid_channels, hid_channels//2,
                               kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels//2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 4*hid_features//hid_channels, 1)
            nn.ConvTranspose2d(hid_channels//2, hid_channels//2,
                               kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels//2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 8*hid_features//hid_channels, 1)
            nn.ConvTranspose2d(hid_channels//2, hid_channels//2,
                               kernel_size=(1, 7), stride=(1, 7), padding=0),
            nn.BatchNorm2d(hid_channels//2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 8*hid_features//hid_channels, 7)
            nn.ConvTranspose2d(hid_channels//2, out_channels,
                               kernel_size=(1, 12), stride=(1, 12), padding=0),
            # output shape: (batch_size, out_channels, 8*hid_features//hid_channels, n_pitches)
            Reshape(shape=[1, 1, self.n_steps_per_bar, self.n_pitches])
            # output shape: (batch_size, out_channels, 1, n_steps_per_bar, n_pitches)
        )
        
    def forward(self, x):
        fx = self.net(x)
        return fx
