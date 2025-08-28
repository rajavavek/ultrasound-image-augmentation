import cv2
import torch
import numpy as np
import random
from scipy.stats import truncnorm
from src.config import cfg

class GANCollator():
    def __init__(self, netG, conditions, weights, channels, sobel, n_to_generate, gan_sampling, generator_batch_size,
                 Y="concentration", conc_intervals=cfg.data.concentration_intervals,
                 input_type="noise", truncated=False, only_fake=True, transforms=None, device=cfg.device):
        netG.to(device)
        netG.eval()
        self.Y = Y
        self.conc_intervals = conc_intervals
        self.netG = netG
        self.conditions = conditions
        self.weights = weights
        self.generator_batch_size = generator_batch_size
        self.channels = channels
        self.sobel = sobel
        self.n_to_generate = n_to_generate
        self.gan_sampling = gan_sampling
        self.truncated = truncated
        self.only_fake = only_fake
        self.input_type = input_type
        self.transforms = transforms
        self.device = device

    def __call__(self, data):
        real_images, real_labels = zip(*data)
        real_images = torch.tensor(np.array(real_images))
        real_labels = torch.tensor(np.array(real_labels))

        if real_labels.shape[1] > 0:
            n_to_generate = self.n_to_generate
        else:
            n_to_generate = len(real_labels)

        batches = [self.generator_batch_size] * (n_to_generate // self.generator_batch_size)
        batches.append(n_to_generate % self.generator_batch_size) if (n_to_generate % self.generator_batch_size) != 0 else None

        gen_imgs = torch.tensor([]).to(self.device)
        gen_labels = torch.tensor([]).to(self.device)
        if self.netG.Y == "concentration":
            if self.gan_sampling == "random":
                min_conc = self.conditions.min().item()
                max_conc = self.conditions.max().item()
                for batch in batches:
                    z = get_input_noise(real_images, batch, self.netG.nz, self.truncated, self.input_type).to(self.device)
                    cond = torch.randint(min_conc, max_conc, (batch, 1), dtype=torch.float32, device=self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])
            
            elif self.gan_sampling == "skewed_random":
                min_conc = self.conditions.min().item()
                max_conc = self.conditions.max().item()
                concentration_values = np.arange(min_conc,max_conc,1)
                for batch in batches:
                    z = get_input_noise(real_images, batch, self.netG.nz, self.truncated, self.input_type).to(self.device)
                    cond = random.choices(concentration_values, weights=1/(concentration_values+max_conc/10), k=batch)
                    cond = torch.tensor(cond).unsqueeze(1).float().to(self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])

            elif self.gan_sampling == "weighted":
                for batch in batches:
                    z = get_input_noise(real_images, batch, self.netG.nz, self.truncated, self.input_type).to(self.device)
                    idx = torch.multinomial(self.weights, batch, replacement=True)
                    cond = self.conditions[idx].unsqueeze(1).float().to(self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])
            
            elif self.gan_sampling == "weighted_interval":
                for batch in batches:
                    z = get_input_noise(real_images, batch, self.netG.nz, self.truncated, self.input_type).to(self.device)
                    idx = torch.multinomial(self.weights, batch, replacement=True)
                    cond = self.conditions[idx].float()
                    vary_cond = (0.75 - 1.25) * torch.rand_like(cond) + 1.25
                    cond = (cond*vary_cond).unsqueeze(1).int().float().to(self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])
            
            elif self.gan_sampling == "weighted_interval_2":
                for batch in batches:
                    z = get_input_noise(real_images, batch, self.netG.nz, self.truncated, self.input_type).to(self.device)
                    #conditions = torch.tensor([0, 5, 9, 20, 100, 400, 800, 3, 7, 15, 50, 200, 600], dtype=torch.float32)
                    conditions = torch.tensor([0,5,10,15,20,30,35,80,113,218,580,1000,12,60,105,160,450], dtype=torch.float32)
                    idx = torch.randint(0, len(conditions), (batch, ))
                    cond = conditions[idx].unsqueeze(1).float().to(self.device)
                    #vary_cond = (0.75 - 1.25) * torch.rand_like(cond) + 1.25
                    #cond = (cond*vary_cond).unsqueeze(1).int().float().to(self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])
            
            elif self.gan_sampling.startswith("restricted"):
                low = int(self.gan_sampling.split("_")[1])
                high = int(self.gan_sampling.split("_")[2])
                for batch in batches:
                    z = get_input_noise(real_images, batch, self.netG.nz, self.truncated, self.input_type).to(self.device)
                    cond = torch.randint(low, high, (batch, 1), dtype=torch.float32, device=self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])

        elif self.netG.Y == "label":
            unique_labels, counts = real_labels.unique(return_counts=True)

            if self.gan_sampling == "balanced":
                samples_per_label = int((len(real_labels) + self.n_to_generate ) / len(unique_labels))
                excess_samples = (len(real_labels) + self.n_to_generate) - (samples_per_label * len(unique_labels))
                extra_sample = random.sample(unique_labels.tolist(), excess_samples)
                all_conds = torch.tensor([])
                for label, count in zip(unique_labels, counts):
                    n_to_generate = samples_per_label - count
                    if label in extra_sample:
                        n_to_generate += 1
                    cond = torch.ones(n_to_generate, dtype=torch.float32, device=self.device)*label
                    all_conds = torch.concat([all_conds, cond])

                prev_batch = 0
                for batch in batches:
                    z = get_input_noise(batch, self.netG.nz, self.truncated).to(self.device)
                    cond = all_conds[prev_batch:batch]
                    prev_batch = batch
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])

            elif self.gan_sampling == "real_frequencies":
                for batch in batches:
                    z = get_input_noise(batch, self.netG.nz, self.truncated).to(self.device)
                    cond = torch.multinomial(self.weights, batch, replacement=True).to(self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])

            elif self.gan_sampling == "random":
                for batch in batches:
                    z = get_input_noise(batch, self.netG.nz, self.truncated).to(self.device)
                    cond = torch.randint(0, self.netG.n_classes, batch, replacement=True).to(self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])

            elif self.gan_sampling == "weighted":
                for batch in batches:
                    z = get_input_noise(batch, self.netG.nz, self.truncated).to(self.device)
                    cond = torch.multinomial(self.weights, batch, replacement=True).to(self.device)
                    gen_imgs = torch.concat([gen_imgs, self.netG(z, cond)])
                    gen_labels = torch.concat([gen_labels, cond])

        if self.sobel:
            gen_imgs = (gen_imgs - gen_imgs.min()) / (gen_imgs.max() - gen_imgs.min())
            gen_imgs = (gen_imgs * 255).cpu().detach().numpy()[:,0,:,:]
            sobel_gen_imgs = np.array([])
            for image in gen_imgs:
                sobel_image = np.expand_dims(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5), (0,1))
                sobel_gen_imgs = np.concatenate([sobel_gen_imgs, sobel_image]) if len(sobel_gen_imgs) > 0 else sobel_image
            gen_imgs = sobel_gen_imgs - np.min(sobel_gen_imgs)
            gen_imgs = gen_imgs / (np.max(gen_imgs)- np.min(sobel_gen_imgs))
        else:
            gen_imgs = (gen_imgs - torch.amin(gen_imgs, dim=(1,2,3), keepdim=True))/(torch.amax(gen_imgs, dim=(1,2,3), keepdim=True) - torch.amin(gen_imgs, dim=(1,2,3), keepdim=True))
            gen_imgs = gen_imgs.cpu().detach().numpy()

        gen_imgs = (gen_imgs-0.5) / 0.5

        if self.transforms:
            transformed_gen_imgs = np.array([])
            for image in gen_imgs:
                data = self.transforms(image=image)
                image = data['image']
                transformed_gen_imgs = np.concatenate([transformed_gen_imgs, image]) if len(transformed_gen_imgs) > 0 else image
            gen_imgs = transformed_gen_imgs

        gen_imgs = np.repeat(gen_imgs, self.channels, axis=1)

        gen_imgs = torch.tensor(gen_imgs)
        gen_labels = gen_labels.cpu()

        if self.Y == "label":
            y=0
            for conc in self.conc_intervals:
                lower_conc, higher_conc = conc.split("-")
                if higher_conc == "inf":
                    higher_conc = np.inf
                else:
                    higher_conc = int(higher_conc)
                gen_labels[(gen_labels >= int(lower_conc)) & (gen_labels <= higher_conc)] = y
                y+=1
            gen_labels = gen_labels.type(torch.LongTensor)
            gen_labels = gen_labels.squeeze(1)

        if self.only_fake:
            images = gen_imgs
            labels = gen_labels
            real_fake_labels = torch.zeros_like(gen_labels, dtype=torch.float32)
        else:
            idx = torch.randperm(real_labels.shape[0] + gen_labels.shape[0])
            images = torch.concat([real_images, gen_imgs])[idx]
            labels = torch.concat([real_labels, gen_labels])[idx]
            real_fake_labels = torch.concat([torch.ones_like(real_labels, dtype=torch.float32), torch.zeros_like(gen_labels, dtype=torch.float32)])[idx]

        return images, labels, real_fake_labels
    
def get_input_noise(real_images, batch, z_dim, truncated=False, input_type="noise"):
    if input_type == "noise":
        if truncated:
            z = torch.tensor(truncnorm.rvs(-1.5, 1.5, size=(batch, z_dim)))
        else:
            z = torch.randn(batch, z_dim)
    elif input_type == "img":
        if len(real_images) >= batch:
            idx = torch.randperm(len(real_images))[:batch]
        else:
            idx = torch.randint(len(real_images), (batch,))
        z = real_images[idx]
    return z

