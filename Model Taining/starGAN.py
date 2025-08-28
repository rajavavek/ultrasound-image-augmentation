import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from src.config import cfg

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class starGAN_Generator(nn.Module):
    """Generator network."""
    def __init__(self, Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), 
                 embedding_dim=cfg.model.embedding_dim, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, ndf=cfg.model.ndf, nc=cfg.model.nc, repeat_num=6):
        super(starGAN_Generator, self).__init__()

        self.Y = Y
        self.ngf = ngf
        self.min_conc = min_conc
        self.max_conc = max_conc

        extra_channel = 1

        layers = []
        layers.append(nn.Conv2d(1+extra_channel, ngf, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(ngf, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = ngf
        for i in range(4):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(4):
            output_padding=(0,0)
            if i==0:
                output_padding = (1,1)
            elif i==1:
                output_padding = (0,1)
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False, output_padding=output_padding))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

        self.embed_cond = nn.Sequential(
                        collections.OrderedDict(
                        [ # input is Z, going into a convolution
                            ("linear0_c", nn.Linear(1, embedding_dim)),
                            ("relu0_c", nn.LeakyReLU(0.2, inplace=True)),
                            ("linear1_c", nn.Linear(embedding_dim, embedding_dim*4)),
                            ("relu1_c", nn.LeakyReLU(0.2, inplace=True)),
                            ("linear2_c", nn.Linear(embedding_dim*4, extra_channel*200*556)),
                            ("unflatten_c", nn.Unflatten(1, (extra_channel, 200, 556)))
                        ]
                    )
                )

    def forward(self, img, cond):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        cond = (cond - self.min_conc)/(self.max_conc - self.min_conc)
        cond_embed = self.embed_cond(cond)
        x = torch.cat([img, cond_embed], dim=1)
        return self.main(x)

class starGAN_Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim=cfg.model.ndf, repeat_num=6):
        super(starGAN_Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        #kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=(3,8), bias=False)
        self.conv2 = nn.Conv2d(curr_dim, 1, kernel_size=(3,8), bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = nn.Flatten()(self.conv1(h))
        out_cond = nn.Flatten()(self.conv2(h))
        return out_src, out_cond
    
