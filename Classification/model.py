from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import collections
from src.config import cfg

def get_classifier(classifier_name=cfg.classifier.name, nf=cfg.classifier.nf, n_classes=len(cfg.data.concentration_intervals)):
    nc=1
    if classifier_name == "cnn_linear":
        return nn.Sequential(
                collections.OrderedDict(
                    [
                        #Input shape: (b, nc, 256, 256)
                        ("conv1", nn.Conv2d(nc, nf, (3,3), padding=1, bias=False)),
                        # ("bn1", nn.BatchNorm2d(nf)),
                        ("leakyrelu1", nn.LeakyReLU(0.2, inplace=True)),
                        ("pool1", nn.MaxPool2d(2, 2)),

                        #Output shape: (b, nf, 128, 128)

                        ("conv2", nn.Conv2d(nf, nf*2, (3,3), padding=1, bias=False)),
                        ("bn2", nn.BatchNorm2d(nf*2)),
                        ("leakyrelu2", nn.LeakyReLU(0.2, inplace=True)),
                        ("pool2", nn.MaxPool2d(2, 2)),

                        #Output shape: (b, nf*2, 64, 64)

                        ("conv3", nn.Conv2d(nf*2, nf*4, (3,3), padding=1, bias=False)),
                        ("bn3", nn.BatchNorm2d(nf*4)),
                        ("leakyrelu3", nn.LeakyReLU(0.2, inplace=True)),
                        ("pool3", nn.MaxPool2d(2, 2)),

                        #Output shape: (b, nf*4, 32, 32)

                        ("conv4", nn.Conv2d(nf*4, nf*8, (3,3), padding=1, bias=False)),
                        ("bn4", nn.BatchNorm2d(nf*8)),
                        ("leakyrelu4", nn.LeakyReLU(0.2, inplace=True)),
                        ("pool4", nn.MaxPool2d(2, 2)),

                        #Output shape: (b, nf*6, 16, 16)

                        ("conv5", nn.Conv2d(nf*8, nf*12, (3,3), padding=1, bias=False)),
                        ("bn5", nn.BatchNorm2d(nf*12)),
                        ("leakyrelu5", nn.LeakyReLU(0.2, inplace=True)),
                        ("pool5", nn.MaxPool2d(2, 2)),

                        #Output shape: (b, nf*8, 8, 8)

                        ("conv6", nn.Conv2d(nf*12, nf*16, (3,3), padding=1, bias=False)),
                        ("bn6", nn.BatchNorm2d(nf*16)),
                        ("leakyrelu6", nn.LeakyReLU(0.2, inplace=True)),
                        ("pool6", nn.MaxPool2d(2, 2)),

                        ("conv7", nn.Conv2d(nf*16, nf*16, (3,8), bias=False)),
                        ("bn7", nn.BatchNorm2d(nf*16)),
                        ("leakyrelu7", nn.LeakyReLU(0.2, inplace=True)),

                        ("flatten7", nn.Flatten()),

                        ("dropout8", nn.Dropout(0.5)),
                        ("linear8", nn.Linear(nf*16, nf*8)),
                        ("leakyrelu8", nn.LeakyReLU(0.2, inplace=True)),
                        
                        ("dropout9", nn.Dropout(0.5)),
                        ("linear9", nn.Linear(nf*8, n_classes))
                    ]
                )    
            )