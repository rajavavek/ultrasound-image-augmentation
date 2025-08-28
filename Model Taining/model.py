from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
import collections
from src.config import cfg
from src.PATHS import *
from .fixed_smooth_conv import UpSampling2d, DownSampling2d

# Generator Code
class Generator(nn.Module):
    def __init__(self, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc):
        super(Generator, self).__init__()
        self.gen_name=gen_name
        self.nz = nz
        self.nc = nc
        self.gen = get_generator(gen_name, nz, ngf, nc)

    def forward(self, input_noise):
        gen_img = self.gen(input_noise)
        #gen_img = nn.functional.interpolate(gen_img, size=(200, 556), mode='area')
        return gen_img
    
class Discriminator(nn.Module):
    def __init__(self, dis_name=cfg.model.dis_name, ndf=cfg.model.ndf, nc=cfg.model.nc):
        super(Discriminator, self).__init__()
        self.dis = get_discriminator(dis_name, ndf, nc)

    def forward(self, input):
        return self.dis(input)
    
class Critic(nn.Module):
    def __init__(self, dis_name=cfg.model.dis_name, ndf=cfg.model.ndf, nc=cfg.model.nc):
        super(Discriminator, self).__init__()
        self.dis = get_discriminator(dis_name, ndf, nc)

    def forward(self, input):
        return self.dis(input)
    
class Encoder(nn.Module):
    def __init__(self, enc_name=cfg.model.dis_name, nz=cfg.model.nz, ndf=cfg.model.ndf, nc=cfg.model.nc):
        super(Encoder, self).__init__()

        #self.hidden = nn.Linear(ngf*32, ngf*32)
        self.mean = nn.Linear(ndf*32, nz)
        self.log_var = nn.Linear(ndf*32, nz)

        self.enc = get_encoder(enc_name, nc, ndf)

    def forward(self, input):

        x = self.enc(input)
        #x = self.hidden(x)
        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, dec_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc):
        super(Decoder, self).__init__()
        self.nz = nz
        self.dec = get_decoder(dec_name, nz, ngf, nc)

    def forward(self, input):
        return self.dec(input)

#Generator with varying stride and reshape
def get_generator(gen_name, nz, ngf, nc):

    if gen_name == "GEN_SQUARE_128":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # # input is Z, going into a linear layer
                        ("linear0", nn.Linear(nz, (ngf*16*4*4))),
                        ("unlatten0", nn.Unflatten(1, ((ngf*16), 4, 4))),

                        # #Output shape: (b, ngf*12, 1, 1)

                        # ("conv1", nn.ConvTranspose2d(ngf*32, ngf*16, (4, 4), 1, 0, bias=False)),
                        # ("bn1", nn.BatchNorm2d(ngf*16)),
                        # ("relu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*12, 4, 4)

                        ("conv1", nn.ConvTranspose2d(ngf*16, ngf*8, (4, 4), 2, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ngf*8)),
                        ("relu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 8, 8)

                        ("conv2", nn.ConvTranspose2d(ngf*8, ngf*4, (4, 4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*4)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 16, 16)

                        ("conv3", nn.ConvTranspose2d(ngf*4, ngf*2, (4, 4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*2)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*6, 32, 32)

                        ("conv4", nn.ConvTranspose2d(ngf*2, ngf, (4, 4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        ("conv5", nn.ConvTranspose2d(ngf, ngf, (4, 4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        ("conv6",nn.Conv2d(ngf, nc, (3, 3), 1, 1, bias=False)),
                        ("tanh6", nn.Tanh())
                    ]
                )
            )

    elif gen_name == "GEN_VARY_KERNEL":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # input is Z, going into a convolution

                        ("linear0", nn.Linear(nz, (ngf*14*3*4))),
                        ("bn0", nn.BatchNorm1d(ngf*14*3*4)),
                        ("relu0", nn.LeakyReLU(0.2, inplace=True)),
                        ("unlatten0", nn.Unflatten(1, ((ngf*14), 3, 4))),

                        # ("conv1", nn.ConvTranspose2d(ngf*16, ngf*12, (3, 4), 1, 0, bias=False)),
                        # ("bn1", nn.BatchNorm2d(ngf*12)),
                        # ("relu1", nn.ReLU(True)),

                        #Output shape: (b, ngf*12, 3, 4)

                        ("conv2", nn.ConvTranspose2d(ngf*14, ngf*12, (3, 5), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*12)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 5, 9)

                        ("conv3", nn.ConvTranspose2d(ngf*12, ngf*10, (2, 4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*10)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 8, 18)

                        ("conv4", nn.ConvTranspose2d(ngf*10, ngf*8, (2, 3), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf*8)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*6, 14, 35)

                        ("conv5", nn.ConvTranspose2d(ngf*8, ngf*6, (2, 4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf*6)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*4, 26, 70)

                        ("conv6",nn.ConvTranspose2d(ngf*6, ngf*4, (2, 3), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ngf*4)),
                        ("relu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*2, 50, 139)

                        ("conv7", nn.ConvTranspose2d(ngf*4, ngf*2, (4, 4), 2, 1, bias=False)),
                        ("bn7", nn.BatchNorm2d(ngf*2)),
                        ("relu7", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf, 100, 278)

                        ("conv8", nn.ConvTranspose2d(ngf*2, ngf, (4, 4), 2, 1, bias=False)),
                        ("bn8", nn.BatchNorm2d(ngf)),
                        ("relu8", nn.LeakyReLU(0.2, inplace=True)),

                        ("conv9", nn.Conv2d(ngf, nc, (3,3), 1, 1, bias=True)),
                        ("tanh9", nn.Tanh())

                        #Output shape: (b, nc, 200, 556)
                    ]
                )
            )
    elif gen_name == "GEN_UPSAMPLE":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # input is Z, going into a convolution
                        # ("conv1", nn.ConvTranspose2d(nz, ngf*12, (3, 4), 1, 0, bias=False)),
                        # ("bn1", nn.BatchNorm2d(ngf*12)),
                        # ("relu1", nn.ReLU(True)),

                        ("linear0", nn.Linear(nz, (ngf*16*4*4))),
                        ("unlatten0", nn.Unflatten(1, ((ngf*16), 4, 4))),

                        #Output shape: (b, ngf*12, 4, 4)

                        ("upsample1", nn.Upsample(size=(6,8), mode='nearest')),
                        ("conv1", nn.Conv2d(ngf*16, ngf*10, (3, 3), 1, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ngf*10)),
                        ("relu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 5, 9)

                        ("upsample2", nn.Upsample(size=(8,16), mode='nearest')),
                        ("conv2", nn.Conv2d(ngf*10, ngf*8, (3, 3), 1, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*8)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 8, 18)

                        ("upsample3", nn.Upsample(size=(14,35), mode='nearest')),
                        ("conv3", nn.Conv2d(ngf*8, ngf*6, (3, 3), 1, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*6)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*6, 14, 35)

                        ("upsample4", nn.Upsample(size=(28,70), mode='nearest')),
                        ("conv4", nn.Conv2d(ngf*6, ngf*4, (3, 3), 1, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf*4)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*4, 26, 70)
                        ("upsample5", nn.Upsample(size=(50,139), mode='nearest')),
                        ("conv5",nn.Conv2d(ngf*4, ngf*2, (3, 3), 1, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf*2)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*2, 50, 139)
                        ("upsample6", nn.Upsample(size=(100,278), mode='nearest')),
                        ("conv6", nn.Conv2d(ngf*2, ngf, (3, 3), 1, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ngf)),
                        ("relu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf, 100, 278)
                        ("upsample7", nn.Upsample(size=(200,556), mode='nearest')),
                        ("conv7", nn.Conv2d(ngf, nc, (3, 3), 1, 1, bias=False)),
                        ("tanh7", nn.Tanh())

                        #Output shape: (b, nc, 200, 556)
                    ]
                )
            )
    elif gen_name == "GEN_VARY_KERNEL_SMOOTH_CONV":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # input is Z, going into a convolution

                        ("linear0", nn.Linear(nz, (ngf*16*3*4))),
                        ("bn0", nn.BatchNorm1d(ngf*16*3*4)),
                        ("relu0", nn.LeakyReLU(0.2, inplace=True)),
                        ("unlatten0", nn.Unflatten(1, ((ngf*16), 3, 4))),

                        # ("conv1", nn.ConvTranspose2d(ngf*16, ngf*12, (3, 4), 1, 0, bias=False)),
                        # ("bn1", nn.BatchNorm2d(ngf*12)),
                        # ("relu1", nn.ReLU(True)),

                        #Output shape: (b, ngf*12, 3, 4)

                        ("upsampling1", UpSampling2d(ngf*16, ngf*10, (3, 5), 2, 1)),
                        ("conv1", nn.Conv2d(ngf*10, ngf*10, (3, 3), 1, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ngf*10)),
                        ("relu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 5, 9)

                        ("upsampling2", UpSampling2d(ngf*10, ngf*8, (2, 4), 2, 1)),
                        ("conv2", nn.Conv2d(ngf*8, ngf*8, (3, 3), 1, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*8)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 8, 18)

                        ("upsampling3", UpSampling2d(ngf*8, ngf*6, (2, 3), 2, 1)),
                        ("conv3", nn.Conv2d(ngf*6, ngf*6, (3, 3), 1, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*6)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*6, 14, 35)

                        ("upsampling4", UpSampling2d(ngf*6, ngf*4, (2, 4), 2, 1)),
                        ("conv4", nn.Conv2d(ngf*4, ngf*4, (3, 3), 1, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf*4)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*4, 26, 70)

                        ("upsampling5", UpSampling2d(ngf*4, ngf*2, (2, 3), 2, 1)),
                        ("conv5", nn.Conv2d(ngf*2, ngf*2, (3, 3), 1, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf*2)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*2, 50, 139)

                        ("upsampling6", UpSampling2d(ngf*2, ngf, (4, 4), 2, 1)),
                        ("conv6", nn.Conv2d(ngf, ngf, (3, 3), 1, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ngf)),
                        ("relu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf, 100, 278)

                        ("upsampling7", UpSampling2d(ngf, ngf, (4, 4), 2, 1)),
                        ("conv7", nn.Conv2d(ngf, ngf, (3, 3), 1, 1, bias=False)),
                        ("bn7", nn.BatchNorm2d(ngf)),
                        ("relu7", nn.LeakyReLU(0.2, inplace=True)),

                        ("conv9", nn.Conv2d(ngf, nc, (3, 3), 1, 1, bias=False)),
                        ("tanh9", nn.Tanh())

                        #Output shape: (b, nc, 200, 556)
                    ]
                )
            )
    elif gen_name == "GEN_VARY_KERNEL_SMOOTH_CONV_HOLD_LAST":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # input is Z, going into a convolution

                        ("linear0", nn.Linear(nz, (ngf*16*3*4))),
                        ("bn0", nn.BatchNorm1d(ngf*16*3*4)),
                        ("relu0", nn.LeakyReLU(0.2, inplace=True)),
                        ("unlatten0", nn.Unflatten(1, ((ngf*16), 3, 4))),

                        # ("conv1", nn.ConvTranspose2d(ngf*16, ngf*12, (3, 4), 1, 0, bias=False)),
                        # ("bn1", nn.BatchNorm2d(ngf*12)),
                        # ("relu1", nn.ReLU(True)),

                        #Output shape: (b, ngf*12, 3, 4)

                        ("conv2", UpSampling2d(ngf*16, ngf*10, (3, 5), 2, 1, hold_mode="hold_last")),
                        ("bn2", nn.BatchNorm2d(ngf*10)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 5, 9)

                        ("conv3", UpSampling2d(ngf*10, ngf*8, (2, 4), 2, 1, hold_mode="hold_last")),
                        ("bn3", nn.BatchNorm2d(ngf*8)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 8, 18)

                        ("conv4", UpSampling2d(ngf*8, ngf*6, (2, 3), 2, 1, hold_mode="hold_last")),
                        ("bn4", nn.BatchNorm2d(ngf*6)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*6, 14, 35)

                        ("conv5", UpSampling2d(ngf*6, ngf*4, (2, 4), 2, 1, hold_mode="hold_last")),
                        ("bn5", nn.BatchNorm2d(ngf*4)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*4, 26, 70)

                        ("conv6", UpSampling2d(ngf*4, ngf*2, (2, 3), 2, 1, hold_mode="hold_last")),
                        ("bn6", nn.BatchNorm2d(ngf*2)),
                        ("relu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*2, 50, 139)

                        ("conv7", UpSampling2d(ngf*2, ngf, (4, 4), 2, 1, hold_mode="hold_last")),
                        ("bn7", nn.BatchNorm2d(ngf)),
                        ("relu7", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf, 100, 278)

                        ("conv8", UpSampling2d(ngf, ngf, (4, 4), 2, 1, hold_mode="hold_last")),
                        ("bn8", nn.BatchNorm2d(ngf)),
                        ("relu8", nn.LeakyReLU(0.2, inplace=True)),

                        ("conv9", nn.Conv2d(ngf, nc, (3, 3), 1, 1, bias=False)),
                        ("tanh9", nn.Tanh())

                        #Output shape: (b, nc, 200, 556)
                    ]
                )
            )
    elif gen_name == "GEN_VARY_STRIDE_RESIZE":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # Input shape: (b, nz, 1, 1)
                        ("conv1", nn.ConvTranspose2d(nz, ngf*10, (4, 4), 1, 0, bias=False)),
                        ("bn1", nn.BatchNorm2d(ngf*10)),
                        ("relu1", nn.ReLU(True)),

                        # Output shape: (b, ngf*10, 4, 4)

                        ("conv2", nn.ConvTranspose2d(ngf*10, ngf*8, (4,4), (2,2), 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*8)),
                        ("relu2", nn.ReLU(True)),

                        # Output shape: (b, ngf*8, 8, 8)

                        ("conv3", nn.ConvTranspose2d(ngf*8, ngf*6, (4,4), (2,3), 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*6)),
                        ("relu3", nn.ReLU(True)),

                        # Output shape: (b, ngf*6, 16, 23)

                        ("conv4", nn.ConvTranspose2d(ngf*6, ngf*4, (4,4), (2,3), 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf*4)),
                        ("relu4", nn.ReLU(True)),

                        # Output shape: (b, ngf*4, 32, 68)

                        ("conv5", nn.ConvTranspose2d(ngf*4, ngf*2, (4,4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf*2)),
                        ("relu5", nn.ReLU(True)),

                        # Output shape: (b, ngf*2, 64, 136)

                        ("conv6", nn.ConvTranspose2d(ngf*2, ngf, (4,4), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ngf)),
                        ("relu6", nn.ReLU(True)),

                        # Output shape: (b, ngf, 128, 272)

                        ("conv7", nn.ConvTranspose2d(ngf, nc, (4,4), 2, 1, bias=False)),
                        ("tanh7", nn.Tanh())

                        # Output shape: (b, nc, 256, 544)
                    
                    ]
                )
            )  
    elif gen_name == "GEN_VARY_NOISE_RESIZE":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        ("linear0", nn.Linear(nz, (ngf*16*3*9))),
                        ("bn0", nn.BatchNorm1d(ngf*16*3*9)),
                        ("relu0", nn.LeakyReLU(0.2, inplace=True)),
                        ("unlatten0", nn.Unflatten(1, ((ngf*16), 3, 9))),

                        #Input shape: (b, nz, 3, 9)
                        ("conv1", nn.ConvTranspose2d(ngf*16, ngf*10, (4,4), 2, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ngf*10)),
                        ("relu1", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ngf*10, 6, 18)

                        ("conv2", nn.ConvTranspose2d(ngf*10, ngf*8, (4,4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*8)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ngf*8, 12, 36)

                        ("conv3", nn.ConvTranspose2d(ngf*8, ngf*6, (4,4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*6)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ngf*6, 24, 72)

                        ("conv4", nn.ConvTranspose2d(ngf*6, ngf*4, (4,4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf*4)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ngf*4, 48, 144)

                        ("conv5", nn.ConvTranspose2d(ngf*4, ngf*2, (4,4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf*2)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*2, 96, 288)

                        ("conv6", nn.ConvTranspose2d(ngf*2, ngf, (4,4), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ngf)),
                        ("relu6", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ngf, 192, 576)

                        ("conv7", nn.Conv2d(ngf, nc, (3,3), 1, 1, bias=False)),
                        ("tanh7", nn.Tanh())

                        # Output shape: (b, nc, 192, 576)
                    ]
                ) 
            )

def get_discriminator(dis_name, ndf, nc):

    if dis_name == "DIS_SQUARE_256":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 256, 256)
                        ("conv1", nn.Conv2d(nc, ndf, (4,4), 2, 1, bias=False)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 128, 128)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 64, 64)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 32, 32)

                        ("conv4", nn.Conv2d(ndf*4, ndf*6, (4,4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ndf*6)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 16, 16)

                        ("conv5", nn.Conv2d(ndf*6, ndf*8, (4,4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 8, 8)

                        ("conv6", nn.Conv2d(ndf*8, ndf*10, (4,4), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ndf*10)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 4, 4)

                        ("conv7", nn.Conv2d(ndf*10, 1, (4,4), 1, 0, bias=False)),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    elif dis_name == "DIS_SQUARE_128":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 256, 256)
                        ("conv1", nn.Conv2d(nc, ndf, (4,4), 2, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ndf)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 128, 128)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 64, 64)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 32, 32)

                        ("conv4", nn.Conv2d(ndf*4, ndf*8, (4,4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 16, 16)

                        ("conv5", nn.Conv2d(ndf*8, ndf*16, (4,4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ndf*16)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 8, 8)

                        # ("conv6", nn.Conv2d(ndf*8, ndf*10, (4,4), 2, 1, bias=False)),
                        # ("bn6", nn.BatchNorm2d(ndf*10)),
                        # ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 4, 4)

                        ("conv6", nn.Conv2d(ndf*16, 1, (4,4), 1, 0, bias=False)),
                        ("flatten6", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    
    elif dis_name == "DIS_SQUARE_128_SN":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 256, 256)
                        ("conv1", spectral_norm(nn.Conv2d(nc, ndf, (4,4), 2, 1, bias=False))),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 128, 128)

                        ("conv2", spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), 2, 1, bias=False))),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 64, 64)

                        ("conv3", spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1, bias=False))),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 32, 32)

                        ("conv4", spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), 2, 1, bias=False))),
                        ("bn4", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 16, 16)

                        ("conv5", spectral_norm(nn.Conv2d(ndf*8, ndf*16, (4,4), 2, 1, bias=False))),
                        ("bn5", nn.BatchNorm2d(ndf*16)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 8, 8)

                        # ("conv6", nn.Conv2d(ndf*8, ndf*10, (4,4), 2, 1, bias=False)),
                        # ("bn6", nn.BatchNorm2d(ndf*10)),
                        # ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 4, 4)

                        ("conv6", spectral_norm(nn.Conv2d(ndf*16, 1, (4,4), 1, 0, bias=False))),
                        ("flatten6", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )

    elif dis_name == "DIS_VARY_KERNEL":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 200, 556)
                        ("conv1", nn.Conv2d(nc, ndf, (4,4), 2, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ndf)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 100, 278)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 50, 139)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (2,3), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 26, 70)

                        ("conv4", nn.Conv2d(ndf*4, ndf*6, (2,4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ndf*6)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 14, 35)

                        ("conv5", nn.Conv2d(ndf*6, ndf*8, (2,3), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 8, 18)

                        ("conv6", nn.Conv2d(ndf*8, ndf*10, (2,4), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ndf*10)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 5, 9)

                        ("conv7", nn.Conv2d(ndf*10, ndf*12, (3,5), 2, 1, bias=False)),
                        ("bn7", nn.BatchNorm2d(ndf*12)),
                        ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*12, 3, 4)

                        ("conv8", nn.Conv2d(ndf*12, 1, (3,4), 1, 0, bias=False)),
                        ("flatten8", nn.Flatten()),


                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    elif dis_name == "DIS_SAME_KERNEL":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 256, 256)
                        ("conv1", nn.Conv2d(nc, ndf, (4,4), 2, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ndf)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 128, 128)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 64, 64)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 32, 32)

                        ("conv4", nn.Conv2d(ndf*4, ndf*8, (4,4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 16, 16)

                        ("conv5", nn.Conv2d(ndf*8, ndf*16, (4,4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ndf*16)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 8, 8)

                        ("conv6", nn.Conv2d(ndf*16, ndf*16, (4,4), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ndf*16)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 4, 4)

                        ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=False)),
                        ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    
    elif dis_name == "DIS_VARY_KERNEL_SMOOTH_CONV":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 200, 556)
                        ("conv1", DownSampling2d(nc, ndf, (4,4), 2, 1)),
                        ("bn1", nn.BatchNorm2d(ndf)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 100, 278)

                        ("conv2", DownSampling2d(ndf, ndf*2, (4,4), 2, 1)),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 50, 139)

                        ("conv3", DownSampling2d(ndf*2, ndf*4, (2,3), 2, 1)),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 26, 70)

                        ("conv4", DownSampling2d(ndf*4, ndf*6, (2,4), 2, 1)),
                        ("bn4", nn.BatchNorm2d(ndf*6)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 14, 35)

                        ("conv5", DownSampling2d(ndf*6, ndf*8, (2,3), 2, 1)),
                        ("bn5", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 8, 18)

                        ("conv6", DownSampling2d(ndf*8, ndf*10, (2,4), 2, 1)),
                        ("bn6", nn.BatchNorm2d(ndf*10)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 5, 9)

                        ("conv7", DownSampling2d(ndf*10, ndf*12, (3,5), 2, 1)),
                        ("bn7", nn.BatchNorm2d(ndf*12)),
                        ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*12, 3, 4)

                        ("conv8", nn.Conv2d(ndf*12, 1, (3,4), 1, 0, bias=False)),
                        ("flatten8", nn.Flatten()),


                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )

    elif dis_name == "DIS_SAME_KERNEL_SMOOTH_CONV":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 256, 256)
                        ("conv1", DownSampling2d(nc, ndf, (4,4), 2, 1)),
                        ("bn1", nn.BatchNorm2d(ndf)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 128, 128)

                        ("conv2", DownSampling2d(ndf, ndf*2, (4,4), 2, 1)),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 64, 64)

                        ("conv3", DownSampling2d(ndf*2, ndf*4, (4,4), 2, 1)),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 32, 32)

                        ("conv4", DownSampling2d(ndf*4, ndf*8, (4,4), 2, 1)),
                        ("bn4", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 16, 16)

                        ("conv5", DownSampling2d(ndf*8, ndf*16, (4,4), 2, 1)),
                        ("bn5", nn.BatchNorm2d(ndf*16)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 8, 8)

                        ("conv6", DownSampling2d(ndf*16, ndf*16, (4,4), 2, 1)),
                        ("bn6", nn.BatchNorm2d(ndf*16)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 4, 4)

                        ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=False)),
                        ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    
    elif dis_name == "DIS_LINEAR":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # Input shape: (b, c, 200, 556)
                        ("conv1", nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ndf, 100, 278)

                        ("conv2", nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ndf * 2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ndf*2, 50, 139)

                        ("conv3", nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ndf * 4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ndf*4, 25, 69)
                        ("conv4", nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ndf * 8)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, ndf*8, 12, 34)

                        ("conv5", nn.Conv2d(ndf*8, 1 , 4, 2, 1, bias=False)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        # Output shape: (b, 1, 6, 17)

                        ("flatten6", nn.Flatten(1,-1)),
                        ("linear6", nn.Linear(6*17, 1)),

                        # Output shape: (b, 1)
                    ]
                )
            )
    
    elif dis_name == "DIS_SAME_KERNEL_LN":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc+1, 200, 556)
                        ("conv1", nn.Conv2d(nc, ndf, (4,4), 2, 1)),
                        # ("bn1", nn.LayerNorm((ndf, 100, 278))),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 100, 278)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1)),
                        ("bn2", nn.LayerNorm((ndf*2, 50, 139))),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 50, 139)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1)),
                        ("bn3", nn.LayerNorm((ndf*4, 25, 69))),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 25, 69)

                        ("conv4", nn.Conv2d(ndf*4, ndf*8, (4,4), 2, 1)),
                        ("bn4", nn.LayerNorm((ndf*8, 12, 34))),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 12, 34)

                        ("conv5", nn.Conv2d(ndf*8, ndf*12, (4,4), 2, 1)),
                        ("bn5", nn.LayerNorm((ndf*12, 6, 17))),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 6, 17)

                        ("conv6", nn.Conv2d(ndf*12, ndf*16, (4,4), 2, 1)),
                        ("bn6", nn.LayerNorm((ndf*16, 3, 8))),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 3, 8)

                        ("conv7", nn.Conv2d(ndf*16, ndf*16, (3,8))),
                        ("bn7", nn.LayerNorm((ndf*16, 1, 1))),
                        ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        ("flatten7", nn.Flatten()),

                        ("dropout8", nn.Dropout(0.4)),
                        ("linear8", nn.Linear(ndf*16, ndf*8)),
                        ("leakyrelu8", nn.LeakyReLU(0.2, inplace=True)),
                        ("dropout9", nn.Dropout(0.4)),
                        ("linear9", nn.Linear(ndf*8, 1))

                        # #Output shape: (b, ndf*10, 3, 8)

                        # ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=True)),
                        # ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
def get_decoder(dec_name, nz, ngf, nc):

    if dec_name == "DEC_SQUARE_256":
        return nn.Sequential(
                collections.OrderedDict(
                    [   
                        # input is Z, going into a linear layer
                        ("linear0", nn.Linear(nz, (ngf*10)*1*1)),
                        ("unlatten0", nn.Unflatten(1, ((ngf*10), 1, 1))),

                        #Output shape: (b, ngf*10, 4, 4)

                        ("conv1", nn.ConvTranspose2d(ngf*10, ngf*10, (4, 4), 1, 0, bias=False)),
                        ("bn1", nn.BatchNorm2d(ngf*10)),
                        ("relu1", nn.ReLU(True)),

                        #Output shape: (b, ngf*12, 4, 4)

                        ("conv2", nn.ConvTranspose2d(ngf*10, ngf*8, (4, 4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*8)),
                        ("relu2", nn.ReLU(True)),

                        #Output shape: (b, ngf*10, 8, 8)

                        ("conv3", nn.ConvTranspose2d(ngf*8, ngf*6, (4, 4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*6)),
                        ("relu3", nn.ReLU(True)),

                        #Output shape: (b, ngf*10, 16, 16)

                        ("conv4", nn.ConvTranspose2d(ngf*6, ngf*4, (4, 4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf*4)),
                        ("relu4", nn.ReLU(True)),

                        #Output shape: (b, ngf*6, 32, 32)

                        ("conv5", nn.ConvTranspose2d(ngf*4, ngf*2, (4, 4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf*2)),
                        ("relu5", nn.ReLU(True)),

                        #Output shape: (b, ngf*4, 64, 64)

                        ("conv6",nn.ConvTranspose2d(ngf*2, ngf, (4, 4), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ngf)),
                        ("relu6", nn.ReLU(True)),

                        #Output shape: (b, ngf*2, 128, 128)

                        ("conv7", nn.ConvTranspose2d(ngf, ngf, (4, 4), 2, 1, bias=False)),
                        ("bn7", nn.BatchNorm2d(ngf)),
                        ("relu7", nn.ReLU(True)),
                        
                        #Output shape: (b, ngf, 256, 256)

                        ("conv8", nn.Conv2d(ngf, nc, 3, 1, 1)),
                        ("sigmoid8", nn.Sigmoid())

                        #Output shape: (b, nc, 256, 256)

                    ]
                )
            )
    elif dec_name == "DEC_SQUARE_64":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # # input is Z, going into a linear layer
                        ("linear0", nn.Linear(nz, (ngf*16*4*4))),
                        ("unlatten0", nn.Unflatten(1, ((ngf*16), 4, 4))),

                        #Output shape: (b, ngf*16, 4, 4)

                        ("conv1", nn.ConvTranspose2d(ngf*16, ngf*8, (4, 4), 1, 0, bias=False)),
                        ("bn1", nn.BatchNorm2d(ngf*8)),
                        ("relu1", nn.ReLU(True)),

                        #Output shape: (b, ngf*12, 4, 4)

                        ("conv2", nn.ConvTranspose2d(ngf*8, ngf*4, (4, 4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*4)),
                        ("relu2", nn.ReLU(True)),

                        #Output shape: (b, ngf*10, 8, 8)

                        ("conv3", nn.ConvTranspose2d(ngf*4, ngf*2, (4, 4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*2)),
                        ("relu3", nn.ReLU(True)),

                        #Output shape: (b, ngf*10, 16, 16)

                        ("conv4", nn.ConvTranspose2d(ngf*2, ngf, (4, 4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf)),
                        ("relu4", nn.ReLU(True)),

                        #Output shape: (b, ngf*6, 32, 32)

                        ("conv5",nn.ConvTranspose2d(ngf, nc, (4, 4), 2, 1, bias=False)),
                        ("tanh5", nn.Tanh())
                        # ("bn5", nn.BatchNorm2d(ngf)),
                        # ("relu5", nn.ReLU(True)),

                        # #Output shape: (b, ngf, 64, 64)

                        # ("conv6", nn.Conv2d(ngf, nc, 3, 1, 1)),
                        # ("sigmoid6", nn.Sigmoid())

                        # #Output shape: (b, nc, 64, 64)

                    ]
                )
            )
    elif dec_name == "DEC_SQUARE_128":
        return nn.Sequential(
            collections.OrderedDict(
                [
                    # # input is Z, going into a linear layer
                    ("linear0", nn.Linear(nz, (ngf*16*4*4))),
                    ("bn0", nn.BatchNorm1d(ngf*16*4*4)),
                    ("relu0", nn.LeakyReLU(0.2, inplace=True)),
                    ("unlatten0", nn.Unflatten(1, ((ngf*16), 4, 4))),

                    # #Output shape: (b, ngf*12, 1, 1)

                    # ("conv1", nn.ConvTranspose2d(ngf*32, ngf*16, (4, 4), 1, 0, bias=False)),
                    # ("bn1", nn.BatchNorm2d(ngf*16)),
                    # ("relu1", nn.LeakyReLU(0.2, inplace=True)),

                    #Output shape: (b, ngf*12, 4, 4)

                    ("conv1", nn.ConvTranspose2d(ngf*16, ngf*8, (4, 4), 2, 1, bias=False)),
                    ("bn1", nn.BatchNorm2d(ngf*8)),
                    ("relu1", nn.LeakyReLU(0.2, inplace=True)),

                    #Output shape: (b, ngf*10, 8, 8)

                    ("conv2", nn.ConvTranspose2d(ngf*8, ngf*4, (4, 4), 2, 1, bias=False)),
                    ("bn2", nn.BatchNorm2d(ngf*4)),
                    ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                    #Output shape: (b, ngf*10, 16, 16)

                    ("conv3", nn.ConvTranspose2d(ngf*4, ngf*2, (4, 4), 2, 1, bias=False)),
                    ("bn3", nn.BatchNorm2d(ngf*2)),
                    ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                    #Output shape: (b, ngf*6, 32, 32)

                    ("conv4", nn.ConvTranspose2d(ngf*2, ngf, (4, 4), 2, 1, bias=False)),
                    ("bn4", nn.BatchNorm2d(ngf)),
                    ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                    ("conv5", nn.ConvTranspose2d(ngf, ngf, (4, 4), 2, 1, bias=False)),
                    ("bn5", nn.BatchNorm2d(ngf)),
                    ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                    ("conv6",nn.Conv2d(ngf, nc, (3, 3), 1, 1, bias=False)),
                    ("tanh6", nn.Tanh())
                ]
            )
        )
    elif dec_name == "DEC_VARY_KERNEL":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        # input is Z, going into a convolution

                        ("linear0", nn.Linear(nz, (ngf*16*3*4))),
                        ("bn0", nn.BatchNorm1d(ngf*16*3*4)),
                        ("relu0", nn.LeakyReLU(0.2, inplace=True)),
                        ("unlatten0", nn.Unflatten(1, ((ngf*16), 3, 4))),

                        # ("conv1", nn.ConvTranspose2d(ngf*16, ngf*12, (3, 4), 1, 0, bias=False)),
                        # ("bn1", nn.BatchNorm2d(ngf*12)),
                        # ("relu1", nn.ReLU(True)),

                        #Output shape: (b, ngf*12, 3, 4)

                        ("conv2", nn.ConvTranspose2d(ngf*16, ngf*10, (3, 5), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*10)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 5, 9)

                        ("conv3", nn.ConvTranspose2d(ngf*10, ngf*8, (2, 4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*8)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 8, 18)

                        ("conv4", nn.ConvTranspose2d(ngf*8, ngf*6, (2, 3), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf*6)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*6, 14, 35)

                        ("conv5", nn.ConvTranspose2d(ngf*6, ngf*4, (2, 4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf*4)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*4, 26, 70)

                        ("conv6",nn.ConvTranspose2d(ngf*4, ngf*2, (2, 3), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ngf*2)),
                        ("relu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*2, 50, 139)

                        ("conv7", nn.ConvTranspose2d(ngf*2, ngf, (4, 4), 2, 1, bias=False)),
                        ("bn7", nn.BatchNorm2d(ngf)),
                        ("relu7", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf, 100, 278)

                        ("conv8", nn.ConvTranspose2d(ngf, nc, (4, 4), 2, 1, bias=False)),
                        ("tanh8", nn.Tanh())

                        #Output shape: (b, nc, 200, 556)
                    ]
                )
            )
    
def get_encoder(enc_name, nc, ndf):

    if enc_name == "ENC_SQUARE_256":
        return nn.Sequential(
                    collections.OrderedDict(
                        [
                            #Input shape: (b, nc, 256, 256)
                            ("conv1", nn.Conv2d(nc, ndf, (4,4), 2, 1, bias=False)),
                            ("bn1", nn.BatchNorm2d(ndf)),
                            ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf, 128, 128)

                            ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1, bias=False)),
                            ("bn2", nn.BatchNorm2d(ndf*2)),
                            ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*2, 64, 64)

                            ("conv3", nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1, bias=False)),
                            ("bn3", nn.BatchNorm2d(ndf*4)),
                            ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*4, 32, 32)

                            ("conv4", nn.Conv2d(ndf*4, ndf*6, (4,4), 2, 1, bias=False)),
                            ("bn4", nn.BatchNorm2d(ndf*6)),
                            ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*6, 16, 16)

                            ("conv5", nn.Conv2d(ndf*6, ndf*8, (4,4), 2, 1, bias=False)),
                            ("bn5", nn.BatchNorm2d(ndf*8)),
                            ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*8, 8, 8)

                            ("conv6", nn.Conv2d(ndf*8, ndf*10, (4,4), 2, 1, bias=False)),
                            ("bn6", nn.BatchNorm2d(ndf*10)),
                            ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*10, 4, 4)
                        ]
                    )
                )
    elif enc_name == "ENC_SQUARE_64":
        return nn.Sequential(
                    collections.OrderedDict(
                        [
                            #Input shape: (b, ndf*2, 64, 64)

                            ("conv1", nn.Conv2d(nc, ndf, (3,3), 2, 1, bias=False)),
                            ("bn1", nn.BatchNorm2d(ndf)),
                            ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*4, 32, 32)

                            ("conv2", nn.Conv2d(ndf, ndf*2, (3,3), 2, 1, bias=False)),
                            ("bn2", nn.BatchNorm2d(ndf*2)),
                            ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*6, 16, 16)

                            ("conv3", nn.Conv2d(ndf*2, ndf*4, (3,3), 2, 1, bias=False)),
                            ("bn3", nn.BatchNorm2d(ndf*4)),
                            ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                            # #Output shape: (b, ndf*8, 8, 8)

                            ("conv4", nn.Conv2d(ndf*4, ndf*8, (3,3), 2, 1, bias=False)),
                            ("bn4", nn.BatchNorm2d(ndf*8)),
                            ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                            # #Output shape: (b, ndf*10, 4, 4)

                            # ("conv5", nn.Conv2d(ndf*8, ndf*16, (4,4), 1, 0, bias=False)),
                            # ("bn5", nn.BatchNorm2d(ndf*16)),
                            # ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                            ("flatten6", nn.Flatten()),
                            # ("linear6", nn.Linear(ndf*16, ndf*32)),
                        ]
                    )
                )
    elif enc_name == "ENC_SQUARE_128":
        return nn.Sequential(
                    collections.OrderedDict(
                        [
                            #Input shape: (b, ndf*2, 64, 64)

                            ("conv1", nn.Conv2d(nc, ndf, (3,3), 2, 1, bias=False)),
                            ("bn1", nn.BatchNorm2d(ndf)),
                            ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*4, 32, 32)

                            ("conv2", nn.Conv2d(ndf, ndf*2, (3,3), 2, 1, bias=False)),
                            ("bn2", nn.BatchNorm2d(ndf*2)),
                            ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                            #Output shape: (b, ndf*6, 16, 16)

                            ("conv3", nn.Conv2d(ndf*2, ndf*4, (3,3), 2, 1, bias=False)),
                            ("bn3", nn.BatchNorm2d(ndf*4)),
                            ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                            # #Output shape: (b, ndf*8, 8, 8)

                            ("conv4", nn.Conv2d(ndf*4, ndf*8, (3,3), 2, 1, bias=False)),
                            ("bn4", nn.BatchNorm2d(ndf*8)),
                            ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                            # #Output shape: (b, ndf*10, 4, 4)

                            ("conv5", nn.Conv2d(ndf*8, ndf*16, (3,3), 2, 1, bias=False)),
                            ("bn5", nn.BatchNorm2d(ndf*16)),
                            ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                            # ("conv6", nn.Conv2d(ndf*16, ndf*32, (4,4), 1, 0, bias=False)),
                            # ("bn6", nn.BatchNorm2d(ndf*32)),
                            # ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                            ("flatten6", nn.Flatten()),
                            
                            ("linear6", nn.Linear(ndf*16*4*4, ndf*32)),
                            ("bn6", nn.BatchNorm1d(ndf*32)),
                            ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True))
                        ]
                    )
                )
    elif enc_name == "ENC_VARY_KERNEL":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 200, 556)
                        ("conv1", nn.Conv2d(nc, ndf, (4,4), 2, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ndf)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 100, 278)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 50, 139)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (2,3), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 26, 70)

                        ("conv4", nn.Conv2d(ndf*4, ndf*6, (2,4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ndf*6)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 14, 35)

                        ("conv5", nn.Conv2d(ndf*6, ndf*8, (2,3), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 8, 18)

                        ("conv6", nn.Conv2d(ndf*8, ndf*10, (2,4), 2, 1, bias=False)),
                        ("bn6", nn.BatchNorm2d(ndf*10)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 5, 9)

                        ("conv7", nn.Conv2d(ndf*10, ndf*12, (3,5), 2, 1, bias=False)),
                        ("bn7", nn.BatchNorm2d(ndf*12)),
                        ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*12, 3, 4)
                        ("flatten8", nn.Flatten()),
                        
                        ("linear8", nn.Linear(ndf*12*3*4, ndf*32)),
                        ("bn8", nn.BatchNorm1d(ndf*32)),
                        ("leakyrelu8", nn.LeakyReLU(0.2, inplace=True))


                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )

