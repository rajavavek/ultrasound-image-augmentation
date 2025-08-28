import torch
import torch.nn as nn
from src.config import cfg
import collections

# Generator Code
class C_Generator(nn.Module):
    def __init__(self, Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), conditioning_mode=cfg.model.gen_conditioning_mode, 
                 embedding_dim=cfg.model.embedding_dim, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc):
        super(C_Generator, self).__init__()
        self.gen_name=gen_name
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        self.Y = Y
        self.min_conc = min_conc
        self.max_conc = max_conc
        self.n_classes = n_classes
        self.conditioning_mode = conditioning_mode

        if conditioning_mode == "channel":
            extra_channel = 8
            self.z_to_img = nn.Sequential(
                    collections.OrderedDict(
                        [ # input is Z, going into a convolution
                            ("linear0_z", nn.Linear(nz, ngf*3*4)),
                            ("relu0_z", nn.LeakyReLU(0.2, inplace=True)),
                            ("linear1_z", nn.Linear(ngf*3*4, ngf*4*3*4)),
                            ("relu1_z", nn.LeakyReLU(0.2, inplace=True)),
                            ("linear2_z", nn.Linear(ngf*4*3*4, ngf*8*3*4)),
                            ("unflatten_z", nn.Unflatten(1, ((ngf*8), 3, 4)))
                        ]
                    )
                )
            if Y == "concentration":
                self.embed_cond = nn.Sequential(
                                    collections.OrderedDict(
                                        [ # input is Z, going into a convolution
                                            ("linear0_c", nn.Linear(1, embedding_dim)),
                                            ("relu0_c", nn.LeakyReLU(0.2, inplace=True)),
                                            ("linear1_c", nn.Linear(embedding_dim, embedding_dim*4)),
                                            ("relu1_c", nn.LeakyReLU(0.2, inplace=True)),
                                            ("linear2_c", nn.Linear(embedding_dim*4, ngf*extra_channel*3*4)),
                                            ("unflatten_c", nn.Unflatten(1, (ngf*extra_channel, 3, 4)))
                                        ]
                                    )
                                )
            elif Y == "label":
                self.embed_cond = nn.Sequential(
                    collections.OrderedDict(
                        [ # input is Z, going into a convolution
                            ("embedding0_l", nn.Embedding(n_classes, embedding_dim)),
                            ("relu0_l", nn.LeakyReLU(0.2, inplace=True)),
                            ("linear1_l", nn.Linear(embedding_dim, embedding_dim*4)),
                            ("relu1_l", nn.LeakyReLU(0.2, inplace=True)),
                            ("linear2_l", nn.Linear(embedding_dim*4, ngf*extra_channel*3*4)),
                            ("unflatten_l", nn.Unflatten(1, (ngf*extra_channel, 3, 4)))
                        ]
                    )
                )
        elif conditioning_mode == "latent":
            extra_channel=0
            self.z_to_img = nn.Sequential(
                    collections.OrderedDict(
                        [ # input is Z, going into a convolution
                            ("linear0_z", nn.Linear(nz+embedding_dim, ngf*3*4)),
                            ("relu0_z", nn.LeakyReLU(0.2, inplace=True)),
                            ("linear1_z", nn.Linear(ngf*3*4, ngf*7*3*4)),
                            ("relu1_z", nn.LeakyReLU(0.2, inplace=True)),
                            ("linear2_z", nn.Linear(ngf*7*3*4, ngf*14*3*4)),
                            ("unflatten_z", nn.Unflatten(1, ((ngf*14), 3, 4)))
                        ]
                    )
                )
            if Y == "concentration":
                self.embed_cond = nn.Sequential(
                                    collections.OrderedDict(
                                        [ # input is Z, going into a convolution
                                            ("linear0_c", nn.Linear(1, embedding_dim)),
                                            ("relu0_c", nn.LeakyReLU(0.2, inplace=True)),
                                            ("linear1_c", nn.Linear(embedding_dim, embedding_dim*4)),
                                            ("relu1_c", nn.LeakyReLU(0.2, inplace=True)),
                                            ("linear2_c", nn.Linear(embedding_dim*4, embedding_dim))
                                        ]
                                    )
                                )
            elif Y == "label":
                self.embed_cond = nn.Sequential(
                                    collections.OrderedDict(
                                    [ # input is Z, going into a convolution
                                        ("embedding0_l", nn.Embedding(n_classes, embedding_dim)),
                                        ("relu0_l", nn.LeakyReLU(0.2, inplace=True)),
                                        ("linear1_l", nn.Linear(embedding_dim, embedding_dim*4)),
                                        ("relu1_l", nn.LeakyReLU(0.2, inplace=True)),
                                        ("linear2_l", nn.Linear(embedding_dim*4, embedding_dim))
                                    ]
                                )
                            )

        self.gen = get_generator(gen_name, extra_channel, ngf, nc)

    def forward(self, input_noise, cond):
        if self.Y == "concentration":
            cond = (cond - self.min_conc)/(self.max_conc - self.min_conc)
            #cond = (cond + 1).log()
        cond_embed = self.embed_cond(cond)
        if self.conditioning_mode == "channel":
            z_img = self.z_to_img(input_noise)
            conditioned_input = torch.concat([z_img, cond_embed], dim=1)
        elif self.conditioning_mode == "latent":
            z_cond = torch.concat([input_noise, cond_embed], dim=1)
            conditioned_input = self.z_to_img(z_cond)

        return self.gen(conditioned_input)
    
class C_Discriminator(nn.Module):
    def __init__(self, Y=cfg.model.Y, use_aux=False, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), 
                 conditioning_mode=cfg.model.dis_conditioning_mode, embedding_dim=cfg.model.embedding_dim, dis_name=cfg.model.dis_name, ndf=cfg.model.ndf, nc=cfg.model.nc):
        super(C_Discriminator, self).__init__()

        self.Y = Y
        self.use_aux = use_aux
        self.conditioning_mode = conditioning_mode
        self.min_conc = min_conc
        self.max_conc = max_conc
        if conditioning_mode == "channel":
            extra_channel = 1
            if dis_name.endswith("CONV"):
                self.dis = nn.Sequential(
                        collections.OrderedDict(
                            [
                                ("conv_dis", nn.Conv2d(ndf*16, 1, kernel_size=(3,8), bias=False)),
                                ("flatten_dis", nn.Flatten())
                            ]
                        )
                    )
            elif dis_name.endswith("PATCH"):
                self.dis = nn.Sequential(
                        collections.OrderedDict(
                            [
                                ("conv_dis", nn.Conv2d(ndf*16, 1, 3, 1, 1, bias=False)),
                            ]
                        )
                    )
            else:
                self.dis = nn.Sequential(
                        collections.OrderedDict(
                            [
                                ("dropout8_dis", nn.Dropout(0.4)),
                                ("linear8_dis", nn.Linear(ndf*16*3*8, ndf*8)),
                                ("leakyrelu8_dis", nn.LeakyReLU(0.2, inplace=True)),
                                ("dropout9_dis", nn.Dropout(0.4)),
                                ("linear9_dis", nn.Linear(ndf*8, 1))
                            ]
                        )
                    )

            if Y == "concentration":
                
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
                
            elif Y == "label":
                self.embed_cond = nn.Sequential(
                                collections.OrderedDict(
                                [ # input is Z, going into a convolution
                                    ("embedding0_l", nn.Embedding(n_classes, embedding_dim)),
                                    ("relu0_l", nn.LeakyReLU(0.2, inplace=True)),
                                    ("linear1_l", nn.Linear(embedding_dim, embedding_dim*4)),
                                    ("relu1_l", nn.LeakyReLU(0.2, inplace=True)),
                                    ("linear2_l", nn.Linear(embedding_dim*4, extra_channel*200*556)),
                                    ("unflatten_l", nn.Unflatten(1, (extra_channel, 200, 556)))
                                ]
                            )
                        )
        elif conditioning_mode == "latent":
            extra_channel = 0
            self.dis = nn.Sequential(
                collections.OrderedDict(
                    [
                        ("dropout8", nn.Dropout(0.4)),
                        ("linear8", nn.Linear(ndf*16+embedding_dim, ndf*8)),
                        ("leakyrelu8", nn.LeakyReLU(0.2, inplace=True)),
                        ("dropout9", nn.Dropout(0.4)),
                        ("linear9", nn.Linear(ndf*8, 1))
                        
                    ]
                )
            )
            if Y == "label":
                self.embed_cond = nn.Sequential(
                                collections.OrderedDict(
                                [ # input is Z, going into a convolution
                                    ("embedding0_l", nn.Embedding(n_classes, embedding_dim)),
                                    ("relu0_l", nn.LeakyReLU(0.2, inplace=True)),
                                    ("linear1_l", nn.Linear(embedding_dim, embedding_dim*4)),
                                    ("relu1_l", nn.LeakyReLU(0.2, inplace=True)),
                                    ("linear2_l", nn.Linear(embedding_dim*4, embedding_dim)),
                                ]
                            )
                        )
            elif Y == "concentration":
                self.embed_cond = nn.Sequential(
                                collections.OrderedDict(
                                [ # input is Z, going into a convolution
                                    ("linear0_c", nn.Linear(1, embedding_dim)),
                                    ("relu0_c", nn.LeakyReLU(0.2, inplace=True)),
                                    ("linear1_c", nn.Linear(embedding_dim, embedding_dim*4)),
                                    ("relu1_c", nn.LeakyReLU(0.2, inplace=True)),
                                    ("linear2_c", nn.Linear(embedding_dim*4, embedding_dim))
                                ]
                            )
                        )

        self.feature_extractor = get_discriminator(dis_name, extra_channel, ndf, nc)

    def forward(self, img, cond):
        if self.Y == "concentration":
            cond = (cond - self.min_conc)/(self.max_conc - self.min_conc)
        cond_embed = self.embed_cond(cond)
        if self.conditioning_mode == "channel":
            conditioned_input = torch.concat([img, cond_embed], dim=1)
            conditioned_features = self.feature_extractor(conditioned_input)
        elif self.conditioning_mode == "latent":
            features = self.feature_extractor(img)
            conditioned_features = torch.concat([features, cond_embed], dim=1)
        return self.dis(conditioned_features)
    
class AC_Discriminator(nn.Module):
    def __init__(self, Y=cfg.model.Y, n_classes=len(cfg.data.concentration_intervals), dis_name=cfg.model.dis_name,
                 ndf=cfg.model.ndf, nc=cfg.model.nc):
        super(AC_Discriminator, self).__init__()

        extra_channel = 0
        self.feature_extractor = get_discriminator(dis_name, extra_channel, ndf, nc)
        if Y == "concentration":
            n_classes=1

        if dis_name.endswith("CONV"):
            self.dis_head = nn.Sequential(
                    collections.OrderedDict(
                        [
                            ("conv_dis", nn.Conv2d(ndf*16, n_classes, kernel_size=(3,8), bias=False)),
                            ("flatten_dis", nn.Flatten())
                        ]
                    )
                )
            self.aux_head = nn.Sequential(
                    collections.OrderedDict(
                        [
                            ("conv_aux", nn.Conv2d(ndf*16, n_classes, kernel_size=(3,8), bias=False)),
                            ("flatten_aux", nn.Flatten())
                        ]
                    )
                )
        else:
            self.dis_head = nn.Sequential(
                    collections.OrderedDict(
                        [
                            ("dropout8_dis", nn.Dropout(0.4)),
                            ("linear8_dis", nn.Linear(ndf*16, ndf*8)),
                            ("leakyrelu8_dis", nn.LeakyReLU(0.2, inplace=True)),
                            ("dropout9_dis", nn.Dropout(0.4)),
                            ("linear9_dis", nn.Linear(ndf*8, 1))
                        ]
                    )
                )
            self.aux_head = nn.Sequential(
                        collections.OrderedDict(
                            [
                                ("dropout8_aux", nn.Dropout(0.4)),
                                ("linear8_aux", nn.Linear(ndf*16, ndf*8)),
                                ("leakyrelu8_aux", nn.LeakyReLU(0.2, inplace=True)),
                                ("dropout9_aux", nn.Dropout(0.4)),
                                ("linear9_aux", nn.Linear(ndf*8, n_classes))
                            ]
                        )
                    )   
            
    def forward(self, img):
        
        features = self.feature_extractor(img)
        
        return self.dis_head(features), self.aux_head(features)

def get_generator(gen_name, extra_channel, ngf, nc):

    if gen_name == "GEN_VARY_KERNEL_BN":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, ngf*14+1, 3, 4)

                        ("conv2", nn.ConvTranspose2d(ngf*16, ngf*12, (3, 5), 2, 1, bias=False)),
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
    
    elif gen_name == "SQUARE_GEN_SAME_KERNEL_BN":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, ngf*14+1, 3, 4)

                        ("conv2", nn.ConvTranspose2d(ngf*(14 + extra_channel), ngf*12, (4, 4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ngf*12)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 5, 9)

                        ("conv3", nn.ConvTranspose2d(ngf*12, ngf*10, (4, 4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ngf*10)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 8, 18)

                        ("conv4", nn.ConvTranspose2d(ngf*10, ngf*8, (4, 4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ngf*8)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*6, 14, 35)

                        ("conv5", nn.ConvTranspose2d(ngf*8, ngf*6, (4, 4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ngf*6)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*4, 26, 70)

                        ("conv6",nn.ConvTranspose2d(ngf*6, ngf*4, (4, 4), 2, 1, bias=False)),
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
    elif gen_name == "GEN_VARY_KERNEL_IN":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, ngf*14+1, 3, 4)

                        ("conv2", nn.ConvTranspose2d(ngf*(8 + extra_channel), ngf*12, (3, 5), 2, 1, bias=False)),
                        ("bn2", nn.InstanceNorm2d(ngf*12)),
                        ("relu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 5, 9)

                        ("conv3", nn.ConvTranspose2d(ngf*12, ngf*10, (2, 4), 2, 1, bias=False)),
                        ("bn3", nn.InstanceNorm2d(ngf*10)),
                        ("relu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*10, 8, 18)

                        ("conv4", nn.ConvTranspose2d(ngf*10, ngf*8, (2, 3), 2, 1, bias=False)),
                        ("bn4", nn.InstanceNorm2d(ngf*8)),
                        ("relu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*6, 14, 35)

                        ("conv5", nn.ConvTranspose2d(ngf*8, ngf*6, (2, 4), 2, 1, bias=False)),
                        ("bn5", nn.InstanceNorm2d(ngf*6)),
                        ("relu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*4, 26, 70)

                        ("conv6",nn.ConvTranspose2d(ngf*6, ngf*4, (2, 3), 2, 1, bias=False)),
                        ("bn6", nn.InstanceNorm2d(ngf*4)),
                        ("relu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf*2, 50, 139)

                        ("conv7", nn.ConvTranspose2d(ngf*4, ngf*2, (4, 4), 2, 1, bias=False)),
                        ("bn7", nn.InstanceNorm2d(ngf*2)),
                        ("relu7", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ngf, 100, 278)

                        ("conv8", nn.ConvTranspose2d(ngf*2, ngf, (4, 4), 2, 1, bias=False)),
                        ("bn8", nn.InstanceNorm2d(ngf)),
                        ("relu8", nn.LeakyReLU(0.2, inplace=True)),

                        ("conv9", nn.Conv2d(ngf, nc, (3,3), 1, 1, bias=True)),
                        ("tanh9", nn.Tanh())

                        #Output shape: (b, nc, 200, 556)
                    ]
                )
            )

def get_discriminator(dis_name, extra_channel, ndf, nc):

    if dis_name == "DIS_SAME_KERNEL_LN":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc+1, 200, 556)
                        ("conv1", nn.Conv2d(nc+extra_channel, ndf, (4,4), 2, 1)),
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

                        # ("conv7", nn.Conv2d(ndf*16, ndf*16, (3,8))),
                        # ("bn7", nn.LayerNorm((ndf*16, 1, 1))),
                        # ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        ("flatten7", nn.Flatten())

                        # #Output shape: (b, ndf*10, 3, 8)

                        # ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=True)),
                        # ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    elif (dis_name == "DIS_SAME_KERNEL_LN_CONV") or (dis_name == "DIS_SAME_KERNEL_LN_PATCH"):
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc+1, 200, 556)
                        ("conv1", nn.Conv2d(nc+extra_channel, ndf, (4,4), 2, 1)),
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

                        # ("conv7", nn.Conv2d(ndf*16, ndf*16, (3,8))),
                        # ("bn7", nn.LayerNorm((ndf*16, 1, 1))),
                        # ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        # ("flatten7", nn.Flatten())

                        # #Output shape: (b, ndf*10, 3, 8)

                        # ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=True)),
                        # ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    elif dis_name == "SQUARE_DIS_SAME_KERNEL_LN":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc+1, 200, 556)
                        ("conv1", nn.Conv2d(nc+extra_channel, ndf, (4,4), 2, 1)),
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

                        ("conv7", nn.Conv2d(ndf*16, ndf*16, (4,9))),
                        ("bn7", nn.LayerNorm((ndf*16, 1, 1))),
                        ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        ("flatten7", nn.Flatten())

                        # #Output shape: (b, ndf*10, 3, 8)

                        # ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=True)),
                        # ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    elif dis_name == "DIS_SAME_KERNEL":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc+1, 200, 556)
                        ("conv1", nn.Conv2d(nc+extra_channel, ndf, (4,4), 2, 1)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 100, 278)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 50, 139)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 25, 69)

                        ("conv4", nn.Conv2d(ndf*4, ndf*8, (4,4), 2, 1)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 12, 34)

                        ("conv5", nn.Conv2d(ndf*8, ndf*12, (4,4), 2, 1)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 6, 17)

                        ("conv6", nn.Conv2d(ndf*12, ndf*16, (4,4), 2, 1)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 3, 8)

                        ("conv7", nn.Conv2d(ndf*16, ndf*16, (3,8))),
                        ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        ("flatten7", nn.Flatten())

                        # #Output shape: (b, ndf*10, 3, 8)

                        # ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=True)),
                        # ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    elif dis_name == "DIS_SAME_KERNEL":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc+1, 200, 556)
                        ("conv1", nn.Conv2d(nc+extra_channel, ndf, (4,4), 2, 1)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 100, 278)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 50, 139)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 25, 69)

                        ("conv4", nn.Conv2d(ndf*4, ndf*8, (4,4), 2, 1)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 12, 34)

                        ("conv5", nn.Conv2d(ndf*8, ndf*12, (4,4), 2, 1)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 6, 17)

                        ("conv6", nn.Conv2d(ndf*12, ndf*16, (4,4), 2, 1)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 3, 8)

                        ("conv7", nn.Conv2d(ndf*16, ndf*16, (3,8))),
                        ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        ("flatten7", nn.Flatten())

                        # #Output shape: (b, ndf*10, 3, 8)

                        # ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=True)),
                        # ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )
    
def get_ac_discriminator(dis_name, ndf, nc):

    if dis_name == "DIS_SAME_KERNEL":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc+1, 200, 556)
                        ("conv1", nn.Conv2d(nc, ndf, (4,4), 2, 1, bias=False)),
                        ("bn1", nn.BatchNorm2d(ndf)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf, 100, 278)

                        ("conv2", nn.Conv2d(ndf, ndf*2, (4,4), 2, 1, bias=False)),
                        ("bn2", nn.BatchNorm2d(ndf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*2, 50, 139)

                        ("conv3", nn.Conv2d(ndf*2, ndf*4, (4,4), 2, 1, bias=False)),
                        ("bn3", nn.BatchNorm2d(ndf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*4, 25, 70)

                        ("conv4", nn.Conv2d(ndf*4, ndf*8, (4,4), 2, 1, bias=False)),
                        ("bn4", nn.BatchNorm2d(ndf*8)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*6, 12, 35)

                        ("conv5", nn.Conv2d(ndf*8, ndf*16, (4,4), 2, 1, bias=False)),
                        ("bn5", nn.BatchNorm2d(ndf*16)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*8, 6, 17)

                        ("conv6", nn.Conv2d(ndf*16, ndf*16, (6,17), bias=False)),
                        ("bn6", nn.BatchNorm2d(ndf*16)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),

                        #Output shape: (b, ndf*10, 3, 8)

                        # ("conv7", nn.Conv2d(ndf*16, 1, (3,8), 1, 0, bias=False)),
                        # ("flatten7", nn.Flatten()),

                        #Output shape: (b, 1, 1, 1)
                    ]
                )    
            )