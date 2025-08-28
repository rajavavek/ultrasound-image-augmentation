from torch.utils.data import Dataset
from src.config import cfg
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import cv2
import torch

class InVitroDataset(Dataset):
    def __init__(self, df=None, Y="label", channels=1, real_augmentation=0, sobel=False,
                  transforms=None, return_img_path=False, return_file=False):
        """
        Args:
            data_path (string): Path to the dataset
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if real_augmentation > 1:
            n = int(real_augmentation-1)
            frac = real_augmentation - int(real_augmentation)
            newdf = df.loc[np.repeat(df.index, 1 + n)].reset_index(drop=True)
            frac_df = df.sample(frac=frac, ignore_index=True)
            df = pd.concat([newdf, frac_df])

        self.df = df
        self.Y = Y
        self.channels = channels
        self.sobel = sobel
        self.transforms = transforms
        self.return_file = return_file
        self.return_img_path= return_img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_path = self.df.loc[idx, "file"]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if self.channels==1 else cv2.IMREAD_COLOR) #decide if 1 channel or 3 channels

        #image = cv2.resize(image, (128,128))
        #apply sobel
        if self.sobel:
            image = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            image = (image - np.min(image))/(np.max(image)-np.min(image))
            #image = image*255

        #image = np.float32(2*((image-np.min(image))/(np.max(image)-np.min(image)))-1)
        y = self.df.loc[idx, self.Y]
        
        if self.Y == "concentration":
            y = np.float32(y)
            y = np.expand_dims(y, axis=0)
            
        if self.transforms:
            data = self.transforms(image=image)
            image = data['image']

        if self.channels==1:
            image = np.expand_dims(image, axis=0)
        if self.channels==3:
            image = image.transpose((2,0,1))
        
        return (image, y) + ((self.df.loc[idx, "file number"],) if self.return_file else ()) + ((img_path,) if self.return_img_path else ())

class Real_Fake_Dataset(Dataset):
    def __init__(self, df, netG, Y="label", channels=1, real_augmentation=0, sobel=False, truncated=False,
                  transforms=None, return_img_path=False, return_file=False, device=cfg.device):
        """
        Args:
            data_path (string): Path to the dataset
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if real_augmentation > 1:
            n = int(real_augmentation-1)
            frac = real_augmentation - int(real_augmentation)
            newdf = df.loc[np.repeat(df.index, 1 + n)].reset_index(drop=True)
            frac_df = df.sample(frac=frac, ignore_index=True)
            df = pd.concat([newdf, frac_df])

        self.df = df
        self.netG = netG
        self.netG.eval()
        for p in self.netG.parameters():
            p.requires_grad = False
        self.Y = Y
        self.channels = channels
        self.sobel = sobel
        self.transforms = transforms
        self.truncated = truncated
        self.return_file = return_file
        self.return_img_path= return_img_path
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_path = self.df.loc[idx, "file"]
        y = self.df.loc[idx, self.Y]
        if self.Y == "concentration":
            y = np.float32(y)
            y = np.expand_dims(y, axis=0)
        if img_path == "noise":
            conc = self.df.loc[idx, "concentration"]
            conc = np.float32(conc)
            conc = np.expand_dims(conc, axis=0)
            if self.truncated:
                z = torch.tensor(truncnorm.rvs(-1.5, 1.5, size=(1, self.netG.nz))).to(self.device)
            else:
                z = torch.randn(1, self.netG.nz).to(self.device)
            
            cond = torch.tensor(conc, dtype=torch.float32).unsqueeze(1).to(self.device)
            image = self.netG(z, cond)[0]

            image = (image - image.min())/(image.max()-image.min())
            image = (image-0.5) / 0.5
            image = image.cpu().numpy()
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if self.channels==1 else cv2.IMREAD_COLOR) #decide if 1 channel or 3 channels
            #image = cv2.resize(image, (128,128))
            #apply sobel
            if self.sobel:
                image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
                image = (image - np.min(image))/ (np.max(image)-np.min(image))
                image = image*255

            image = np.float32(2*((image-np.min(image))/(np.max(image)-np.min(image)))-1)
            if self.transforms:
                data = self.transforms(image=image)
                image = data['image']

            if self.channels==1:
                image = np.expand_dims(image, axis=0)
            if self.channels==3:
                image = image.transpose((2,0,1))
        
        return (image, y) + ((self.df.loc[idx, "file number"],) if self.return_file else ()) + ((img_path,) if self.return_img_path else ())

class FakeDataset(Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        return ([], [])
    
class GeneratorDataset(Dataset):
    def __init__(self, G, z_dim, weights, n_images, conditional, channels, sobel, transforms=None, device=cfg.device):
        self.G = G.to(device)
        self.G.eval()
        self.z_dim = z_dim
        self.weights = weights
        self.n_images = n_images
        self.conditional = conditional
        self.channels = channels
        self.sobel = sobel
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return self.n_images

    def __getitem__(self, index):

        z = torch.randn(1, self.z_dim).to(self.device)

        if self.conditional:
            cond = torch.multinomial(self.weights, 1, replacement=True).to(self.device)
            image = self.G(z, cond)[0][0]
        else:
            image = self.G(z)[0]

        if self.sobel:
            image = (((image+1)/2) * 255).cpu().detach().numpy()
            image = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
            image = image - np.min(image)
            image = image / np.max(image)
        else:
            image = image - image.min()
            image = image / image.max()
            image = image.cpu().detach().numpy()
        
        image = (image-0.5)/0.5

        if self.transforms:
            data = self.transforms(image=image)
            image = data['image']

        image = np.repeat(image[np.newaxis, :, :], self.channels, axis=0)

        return image, cond.item() if self.conditional else image