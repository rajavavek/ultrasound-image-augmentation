import torch
from torch import nn
import torchvision.utils as vutils
import pandas as pd
import numpy as np
import itertools
import wandb
from pytorch_msssim import SSIM
from .utils import save_model, load_model
from .model import Generator, Discriminator, Encoder, Decoder
from .conditional_model import C_Generator, C_Discriminator, AC_Discriminator
from .starGAN import starGAN_Generator, starGAN_Discriminator
from .PGGAN import PGGAN_Generator, PGGAN_Discriminator
from src.data.dataloaders import prepare_train_loader
from src.utils import set_seed, flatten_dict
from src.config import cfg
from src.PATHS import *
import hydra

@hydra.main(config_path=PROJECT_PATH.as_posix(), config_name="config", version_base=None)
def train_GAN(cfg):

    set_seed()
    if cfg.model.name.startswith("C-"):
        if cfg.model.name.endswith("PGGAN"):
            model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / cfg.model.name
            netG = PGGAN_Generator(Y=cfg.model.Y, ngf=cfg.model.ngf, latent_size=cfg.model.nz, 
                                   min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, embedding_dim=cfg.model.embedding_dim)
            netD = PGGAN_Discriminator(ndf=cfg.model.ndf)
        else:
            model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / f"{cfg.model.name}_{cfg.model.gen_conditioning_mode}_{cfg.model.dis_conditioning_mode}" / (cfg.model.gen_name + "-" + cfg.model.dis_name)
            netG = C_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), conditioning_mode=cfg.model.gen_conditioning_mode,
                            embedding_dim=cfg.model.embedding_dim, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
            netD = C_Discriminator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), conditioning_mode=cfg.model.dis_conditioning_mode, 
                                embedding_dim=cfg.model.embedding_dim, dis_name=cfg.model.dis_name, ndf=cfg.model.ndf, nc=cfg.model.nc)
        conditional = True
    elif cfg.model.name.startswith("AC"):
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / f"{cfg.model.name}_{cfg.model.gen_conditioning_mode}_{cfg.model.dis_conditioning_mode}" / (cfg.model.gen_name + "-" + cfg.model.dis_name)
        netG = C_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), conditioning_mode=cfg.model.gen_conditioning_mode, 
                           embedding_dim=cfg.model.embedding_dim, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
        netD = AC_Discriminator(Y=cfg.model.Y, n_classes=len(cfg.data.concentration_intervals), dis_name=cfg.model.dis_name,
                                ndf=cfg.model.ndf, nc=cfg.model.nc)
        conditional = True
    elif cfg.model.name == "starGAN":
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / cfg.model.name
        netG = starGAN_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), 
                                 embedding_dim=cfg.model.embedding_dim, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, ndf=cfg.model.ndf, nc=cfg.model.nc)
        netD = starGAN_Discriminator(conv_dim=cfg.model.ndf)
    else:
        netG = Generator(gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
        netD = Discriminator(dis_name=cfg.model.dis_name, ndf=cfg.model.ndf, nc=cfg.model.nc)
        conditional = False

    model_path.mkdir(parents=True, exist_ok=True)

    if not cfg.model.name.endswith("PGGAN"):
        netG.apply(weights_init)
        netD.apply(weights_init)
    netG.to(cfg.device)
    netD.to(cfg.device)

    train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, height=cfg.data.height, width=cfg.data.width,
                                            train_concentrations=cfg.data.train_concentrations, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                                            Y=cfg.model.Y, channels=cfg.data.channels, sobel=cfg.data.sobel, real_augmentation_factor=cfg.classifier.real_augmentation, 
                                            conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size, apply_transforms=True)

    optimizerG = get_optimizer(netG, lr=cfg.hyperparameters.gen_lr, 
                               beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2)
    optimizerD = get_optimizer(netD, lr=cfg.hyperparameters.dis_lr, 
                               beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2)

    if cfg.model.name.endswith("DCGAN"):
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "dcgan", "regressor", "evaluation", "seed", "device"]}
        wandb.init(project="gan", entity="nbs-gan", config=flatten_dict(config))
        wandb.run.log_code(root=SRC_PATH.as_posix(), include_fn=lambda path: path.split("/")[-2]=="model")

        criterion = nn.BCEWithLogitsLoss()

        if cfg.model.name.startswith("AC"):
            netG = ACDCGAN_training(netG, netD, train_dataloader, criterion, optimizerG, optimizerD, model_path, 
                                    n_classes=len(cfg.data.concentration_intervals), condition_sampling=cfg.model.condition_sampling,
                                    epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label, max_real_label=cfg.dcgan.max_real_label,
                                    min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, 
                                    p_noisy_label=cfg.dcgan.p_noisy_label, save_generator_training_output=True, device=cfg.device)

        else:
            netG = DCGAN_training(netG, netD, train_dataloader, criterion, optimizerG, optimizerD, model_path, conditional,
                                    epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label, max_real_label=cfg.dcgan.max_real_label,
                                    min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, 
                                    p_noisy_label=cfg.dcgan.p_noisy_label, save_generator_training_output=True, device=cfg.device)
        
    elif cfg.model.name.endswith("LSGAN"):
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "dcgan", "regressor", "evaluation", "seed", "device"]}
        wandb.init(project="gan", entity="nbs-gan", config=flatten_dict(config))
        wandb.run.log_code(root=SRC_PATH.as_posix(), include_fn=lambda path: path.split("/")[-2]=="model")

        criterion = nn.MSELoss()

        if cfg.model.name.startswith("AC"):
            netG = ACLSGAN_training(netG, netD, train_dataloader, criterion, optimizerG, optimizerD, model_path, conditional, 
                                    condition_sampling=cfg.model.condition_sampling, epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label,
                                    max_real_label=cfg.dcgan.max_real_label, min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, 
                                    p_noisy_label=cfg.dcgan.p_noisy_label, save_generator_training_output=True, device=cfg.device)
        else:
            netG = LSGAN_training(netG, netD, train_dataloader, criterion, optimizerG, optimizerD, model_path, conditional,
                                    epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label, max_real_label=cfg.dcgan.max_real_label,
                                    min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, 
                                    p_noisy_label=cfg.dcgan.p_noisy_label, save_generator_training_output=True, device=cfg.device)
        
        
    elif cfg.model.name.endswith("WGAN"):
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "wgan", "regressor", "evaluation", "seed", "device"]}
        wandb.init(project="gan", entity="nbs-gan", config=flatten_dict(config))
        wandb.run.log_code(root=SRC_PATH.as_posix(), include_fn=lambda path: path.split("/")[-2]=="model")
        if cfg.model.name.startswith("AC"):
            netG = ACWGAN_training(netG, netD, train_dataloader, optimizerG, optimizerD, model_path,
                                   n_classes=len(cfg.data.concentration_intervals), condition_sampling=cfg.model.condition_sampling,
                                   epochs=cfg.hyperparameters.epochs, critic_iters=cfg.wgan.critic_iters, lambda_1=cfg.wgan.lambda_1,
                                   lambda_2=cfg.wgan.lambda_2, save_generator_training_output=True, device=cfg.device)
        else:
            netG = WGAN_training(netG, netD, train_dataloader, optimizerG, optimizerD, model_path, conditional,
                                 n_classes=len(cfg.data.concentration_intervals), condition_sampling=cfg.model.condition_sampling,
                                 epochs=cfg.hyperparameters.epochs, critic_iters=cfg.wgan.critic_iters, lambda_1=cfg.wgan.lambda_1,
                                 lambda_2=cfg.wgan.lambda_2, save_generator_training_output=True, device=cfg.device)     
    elif cfg.model.name == "starGAN":
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "stargan", "regressor", "evaluation", "seed", "device"]}
        wandb.init(project="gan", entity="nbs-gan", config=flatten_dict(config))
        wandb.run.log_code(root=SRC_PATH.as_posix(), include_fn=lambda path: path.split("/")[-2]=="model")

        netG = starGAN_training(netG, netD, train_dataloader, optimizerG, optimizerD, model_path, 
                                epochs=cfg.hyperparameters.epochs, critic_iters=cfg.stargan.critic_iters, lambda_1=cfg.stargan.lambda_1,
                                lambda_cond=cfg.stargan.lambda_cond, lambda_recons=cfg.stargan.lambda_recons,
                                lambda_gdl=cfg.stargan.lambda_gdl, save_generator_training_output=True, device=cfg.device)
    elif cfg.model.name.endswith("PGGAN"):
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "pggan", "regressor", "evaluation", "seed", "device"]}
        wandb.init(project="gan", entity="nbs-gan", config=flatten_dict(config))
        wandb.run.log_code(root=SRC_PATH.as_posix(), include_fn=lambda path: path.split("/")[-2]=="model")

        netG = PGGAN_training(cfg, netG, netD, optimizerG, optimizerD, model_path, conditional,
                                epochs=cfg.hyperparameters.epochs, schedule_start_epochs=cfg.pggan.start_epochs, schedule_batch_size=cfg.pggan.batch_sizes, 
                                schedule_num_epochs=cfg.pggan.num_epochs, critic_iters=cfg.pggan.critic_iters, lambda_1=cfg.pggan.lambda_1,
                                save_generator_training_output=True, device=cfg.device)

    save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")

@hydra.main(config_path=PROJECT_PATH.as_posix(), config_name="config", version_base=None)
def train_VAE(cfg):

    model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / cfg.model.name / (cfg.model.gen_name + "-" + cfg.model.dis_name)
    model_path.mkdir(parents=True, exist_ok=True)

    set_seed()

    netEnc = Encoder(nz=cfg.model.nz, ndf=cfg.model.ndf, nc=cfg.model.nc)
    netEnc.apply(weights_init)
    netEnc.to(cfg.device)

    netDec = Decoder(nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
    netDec.apply(weights_init)
    netDec.to(cfg.device)

    all_params = itertools.chain(netEnc.parameters(), netDec.parameters())
    optimizer = torch.optim.Adam(all_params, lr=cfg.hyperparameters.gen_lr, 
                                 betas=(cfg.hyperparameters.beta1, cfg.hyperparameters.beta2))

    train_dataloader = prepare_train_loader(split=cfg.data.split, augmented=cfg.data.augmented, Y=cfg.model.Y,
                                            channels=cfg.data.channels, conc_intervals=cfg.data.concentration_intervals,
                                            batch_size=cfg.hyperparameters.batch_size)

    config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "seed", "device"]}
    wandb.init(project="vae", entity="nbs-gan", config=flatten_dict(config))
    netEnc, netDec = VAE_training(netEnc, netDec, train_dataloader, optimizer, model_path, 
                                  epochs=cfg.hyperparameters.epochs, save_generator_training_output=True, device=cfg.device)
    save_model(netDec.cpu(), model_path)
    wandb.finish()

@hydra.main(config_path=PROJECT_PATH.as_posix(), config_name="config", version_base=None)
def train_VAE_GAN(cfg):

    model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / cfg.model.name / (cfg.model.gen_name + "-" + cfg.model.dis_name)
    model_path.mkdir(parents=True, exist_ok=True)

    set_seed()
    netG = Decoder(nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
    netG = load_model(netG, "/zhome/69/4/164378/nbs/models/TOY_DATASET/VAE/DEC_SQUARE_128-ENC_SQUARE_128/model.pt")
    netG.to(cfg.device)

    netD = Discriminator(dis_name=cfg.model.dis_name, ndf=cfg.model.ndf, nc=cfg.model.nc)
    netD.apply(weights_init)
    netD.to(cfg.device)

    train_dataloader = prepare_train_loader(split=cfg.data.split, augmented=cfg.data.augmented, Y=cfg.model.Y,
                                            channels=cfg.data.channels, conc_intervals=cfg.data.concentration_intervals,
                                            batch_size=cfg.hyperparameters.batch_size)

    optimizerG = get_optimizer(netG, lr=cfg.hyperparameters.gen_lr,
                                beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2)
    optimizerD = get_optimizer(netD, lr=cfg.hyperparameters.dis_lr,
                                beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2)
    
    if cfg.model.name.endswith("DCGAN"):
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "dcgan", "seed", "device"]}
        wandb.init(project="gan", entity="nbs-gan", config=flatten_dict(config))

        criterion = get_loss_function()

        netG, netD = DCGAN_training(netG, netD, train_dataloader, criterion, optimizerG, optimizerD, model_path, 
                                    epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label, max_real_label=cfg.dcgan.max_real_label,
                                    min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, 
                                    p_noisy_label=cfg.dcgan.p_noisy_label, save_generator_training_output=True, device=cfg.device)
    
    elif cfg.model.name.endswith("WGAN"):
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "wgan", "seed", "device"]}
        wandb.init(project="gan", entity="nbs-gan", config=flatten_dict(config))

        netG, netD = WGAN_training(netG, netD, train_dataloader, optimizerG, optimizerD, model_path,
                                    epochs=cfg.hyperparameters.epochs, critic_iters=cfg.wgan.critic_iters, lambda_1=cfg.wgan.lambda_1,
                                    save_generator_training_output=True, device=cfg.device)

    save_model(netG.cpu(), model_path)
    wandb.finish()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_loss_function():
    # Initialize BCELoss function
    criterion = nn.BCEWithLogitsLoss()
    return criterion

def get_optimizer(net, lr=cfg.hyperparameters.gen_lr, beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2):
    # Setup Adam optimizers for net
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizer

def save_generator_images(epoch, netG, fixed_noise, model_path, conditional, cond=None):

    netG.eval()
    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fake = netG(fixed_noise, cond).detach().cpu() if conditional else netG(fixed_noise).detach().cpu()
        images_grid = vutils.make_grid(fake, padding=2, normalize=True, value_range=(-1,1))

        images_path = model_path / "generated_images"
        images_path.mkdir(parents=True, exist_ok=True)

        if netG.Y == "label":
            img_path = images_path / (f"epoch_{epoch}_label_{cond[0].item()}.png" if conditional else f"epoch_{epoch}.png")
        elif netG.Y == "concentration":
            img_path = images_path / (f"epoch_{epoch}_conc_{int(cond[0].item())}.png" if conditional else f"epoch_{epoch}.png")
        vutils.save_image(images_grid, img_path)

    netG.train()

def save_decoded_images(epoch, netG, z, model_path):
    if epoch % 10 == 0:
        netG.eval()
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(z).detach().cpu()
            #fake = A.Normalize(mean=(-1), std=(2), max_pixel_value=1)(image=np.array(fake))["image"]
            images_grid = vutils.make_grid(fake, padding=2, normalize=True, value_range=(-1,1))

            images_path = model_path / "generated_images"
            images_path.mkdir(parents=True, exist_ok=True)

            img_path = images_path / f"epoch_{epoch}_decoded.png"
            vutils.save_image(images_grid, img_path)

        netG.train()

def save_starGAN_images(netG, fixed_img, cond, epoch, model_path):
    netG.eval()
    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fake = netG(fixed_img, cond).detach().cpu()
        images_grid = vutils.make_grid(fake, padding=2, normalize=True, value_range=(-1,1))

        images_path = model_path / "generated_images"
        images_path.mkdir(parents=True, exist_ok=True)

        if netG.Y == "label":
            img_path = images_path / f"epoch_{epoch}_label_{cond[0].item()}.png"
        elif netG.Y == "concentration":
            img_path = images_path / f"epoch_{epoch}_conc_{int(cond[0].item())}.png"
        vutils.save_image(images_grid, img_path)
    netG.train()

def DCGAN_training(netG, netD, dataloader, criterion, optimizerG, optimizerD, model_path, conditional,
                    epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label, max_real_label=cfg.dcgan.max_real_label,
                    min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, p_noisy_label=cfg.dcgan.p_noisy_label,
                    save_generator_training_output=True, device=cfg.device):
    
    netG.train()
    netD.train()
    min_g_loss = np.inf

    concentrations = dataloader.dataset.df[netG.Y].unique().astype(np.float32)

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, netG.nz, device=device) #noise_h, noise_w

    # For each epoch
    gen_iters = 0
    for epoch in range(epochs):
        # For each batch in the dataloader
        G_losses = []
        D_losses = []
        D_x_list = []
        D_G_z1_list = []
        D_G_z2_list = []

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ############################
            if gen_iters < 10 or gen_iters % 500 == 0:
                critic_iterations = 50
            else:
                critic_iterations = 5
            j=0
            while j < critic_iterations and i < len(dataloader):
                j += 1
                data = next(data_iter)
                i += 1
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                if conditional:
                    real_cond = data[1].to(device)
                    fake_cond = real_cond

                real_label = torch.zeros(b_size, device=device).uniform_(min_real_label, max_real_label)
                n_noisy = int(b_size*p_noisy_label)
                idx = torch.randint(b_size, (n_noisy,))
                try:
                    real_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_fake_label, max_fake_label)
                except:
                    pass
                real_label = real_label.unsqueeze(1)

                # Forward pass real batch through D
                real_output = netD(real_cpu, real_cond) if conditional else netD(real_cpu)
                # Calculate loss on all-real batch
                errD_real = criterion(real_output, real_label)
                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, netG.nz, device=device) #noise_h, noise_w
                # Generate fake image batch with G
                fake = netG(noise, fake_cond) if conditional else netG(noise)

                fake_label = torch.zeros(b_size, device=device).uniform_(min_fake_label, max_fake_label)
                n_noisy = int(b_size*p_noisy_label)
                idx = torch.randint(b_size, (n_noisy,))
                try:
                    fake_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_real_label, max_real_label)
                except:
                    pass
                fake_label = fake_label.unsqueeze(1)

                # Classify all fake batch with D
                fake_output = netD(fake.detach(), fake_cond) if conditional else netD(fake.detach())
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(fake_output, fake_label)
                errD = errD_real + errD_fake

                netD.zero_grad()
                errD.backward()
                # Update D
                optimizerD.step()

                D_x = nn.Sigmoid()(real_output).mean().item()
                D_G_z1 = nn.Sigmoid()(fake_output).mean().item()

                D_losses.append(errD.item())
                D_x_list.append(D_x)
                D_G_z1_list.append(D_G_z1)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            # Since we just updated D, perform another forward pass of all-fake batch through D
            # Generate batch of latent vectors
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise = torch.randn(dataloader.batch_size, cfg.model.nz, device=device)
            if b_size != dataloader.batch_size:
                rand_idx = torch.randint(len(concentrations), (dataloader.batch_size,))
                fake_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)
            # Generate fake image batch with G
            fake = netG(noise, fake_cond) if conditional else netG(noise)
            fake_output_2 = netD(fake, fake_cond) if conditional else netD(fake)
            # Calculate G's loss based on this output
            real_label = torch.zeros(dataloader.batch_size, device=device).uniform_(min_real_label, max_real_label)
            n_noisy = int(dataloader.batch_size*p_noisy_label)
            idx = torch.randint(dataloader.batch_size, (n_noisy,))
            try:
                real_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_fake_label, max_fake_label)
            except:
                pass
            real_label = real_label.unsqueeze(1)
            errG = criterion(fake_output_2, real_label) # fake labels are real for generator cost
            # Calculate gradients for G
            netG.zero_grad()
            errG.backward()
            # Update G
            optimizerG.step()

            gen_iters += 1

            D_G_z2 = nn.Sigmoid()(fake_output_2).mean().item()

            G_losses.append(errG.item())
            D_G_z2_list.append(D_G_z2)

            for p in netD.parameters():
                p.requires_grad = True

        G_loss = np.mean(G_losses)
        D_loss = np.mean(D_losses)
        D_x = np.mean(D_x_list)
        D_G_z1 = np.mean(D_G_z1_list)
        D_G_z2 = np.mean(D_G_z2_list)

        if G_loss < min_g_loss:
            min_g_loss = G_loss
            save_model(netG.cpu(), model_path, f"min_loss_{wandb.run.name}.pt")
            netG.to(device)

        logging_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2,
            }

        wandb.log(logging_dict)

        if epoch % 50 == 0:
            if save_generator_training_output:
                if conditional:
                    if netG.Y == "label":
                        for l in range(netG.n_classes):
                            cond = torch.ones(64, dtype=int, device=device)*l
                            save_generator_images(epoch, netG, fixed_noise, model_path, conditional, cond)
                    elif netG.Y == "concentration":
                        for conc in concentrations:
                            conc = torch.ones((64,1), dtype=torch.float32, device=device)*conc
                            save_generator_images(epoch, netG, fixed_noise, model_path, conditional, conc)
                else:
                    save_generator_images(epoch, netG, fixed_noise, model_path, conditional)

            save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")
            
            netG.to(device)

    return netG

def LSGAN_training(netG, netD, dataloader, criterion, optimizerG, optimizerD, model_path, conditional,
                    epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label, max_real_label=cfg.dcgan.max_real_label,
                    min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, p_noisy_label=cfg.dcgan.p_noisy_label,
                    save_generator_training_output=True, device=cfg.device):
    
    netG.train()
    netD.train()
    min_g_loss = np.inf

    concentrations = dataloader.dataset.df[netG.Y].unique().astype(np.float32)

    # Training Loop
    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, netG.nz, device=device) #noise_h, noise_w

    # For each epoch
    gen_iters = 0
    for epoch in range(epochs):
        # For each batch in the dataloader
        G_losses = []
        D_losses = []
        D_x_list = []
        D_G_z1_list = []
        D_G_z2_list = []

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ############################
            if gen_iters < 10 or gen_iters % 500 == 0:
                critic_iterations = 50
            else:
                critic_iterations = 5
            j=0
            while j < critic_iterations and i < len(dataloader):
                j += 1
                data = next(data_iter)
                i += 1
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                if conditional:
                    real_cond = data[1].to(device)
                    fake_cond = real_cond

                real_label = torch.zeros(b_size, device=device).uniform_(min_real_label, max_real_label)
                n_noisy = int(b_size*p_noisy_label)
                idx = torch.randint(b_size, (n_noisy,))
                try:
                    real_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_fake_label, max_fake_label)
                except:
                    pass
                real_label = real_label.unsqueeze(1)

                # Forward pass real batch through D
                real_output = netD(real_cpu, real_cond) if conditional else netD(real_cpu)
                # Calculate loss on all-real batch
                errD_real = 0.5 * torch.mean((real_output-real_label)**2)
                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, netG.nz, device=device) #noise_h, noise_w
                # Generate fake image batch with G
                fake = netG(noise, fake_cond) if conditional else netG(noise)

                fake_label = torch.zeros(b_size, device=device).uniform_(min_fake_label, max_fake_label)
                n_noisy = int(b_size*p_noisy_label)
                idx = torch.randint(b_size, (n_noisy,))
                try:
                    fake_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_real_label, max_real_label)
                except:
                    pass
                fake_label = fake_label.unsqueeze(1)

                # Classify all fake batch with D
                fake_output = netD(fake.detach(), fake_cond) if conditional else netD(fake.detach())
                # Calculate D's loss on the all-fake batch
                errD_fake = 0.5 * torch.mean((fake_output-fake_label)**2)
                errD = errD_real + errD_fake

                netD.zero_grad()
                errD.backward()
                # Update D
                optimizerD.step()

                D_x = real_output.mean().item()
                D_G_z1 = fake_output.mean().item()

                D_losses.append(errD.item())
                D_x_list.append(D_x)
                D_G_z1_list.append(D_G_z1)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            # Since we just updated D, perform another forward pass of all-fake batch through D
            # Generate batch of latent vectors
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise = torch.randn(dataloader.batch_size, cfg.model.nz, device=device)
            if b_size != dataloader.batch_size:
                rand_idx = torch.randint(len(concentrations), (dataloader.batch_size,))
                fake_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)
            # Generate fake image batch with G
            fake = netG(noise, fake_cond) if conditional else netG(noise)
            fake_output_2 = netD(fake, fake_cond) if conditional else netD(fake)
            # Calculate G's loss based on this output
            real_label = torch.zeros(dataloader.batch_size, device=device).uniform_(min_real_label, max_real_label)
            n_noisy = int(dataloader.batch_size*p_noisy_label)
            idx = torch.randint(dataloader.batch_size, (n_noisy,))
            try:
                real_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_fake_label, max_fake_label)
            except:
                pass
            real_label = real_label.unsqueeze(1)
            errG = 0.5 * torch.mean((fake_output_2-real_label)**2) # fake labels are real for generator cost
            # Calculate gradients for G
            netG.zero_grad()
            errG.backward()
            # Update G
            optimizerG.step()

            gen_iters += 1

            D_G_z2 = fake_output_2.mean().item()

            G_losses.append(errG.item())
            D_G_z2_list.append(D_G_z2)

            for p in netD.parameters():
                p.requires_grad = True

        G_loss = np.mean(G_losses)
        D_loss = np.mean(D_losses)
        D_x = np.mean(D_x_list)
        D_G_z1 = np.mean(D_G_z1_list)
        D_G_z2 = np.mean(D_G_z2_list)

        if G_loss < min_g_loss:
            min_g_loss = G_loss
            save_model(netG.cpu(), model_path, f"min_loss_{wandb.run.name}.pt")
            netG.to(device)

        logging_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2,
            }

        wandb.log(logging_dict)

        if epoch % 50 == 0:
            if save_generator_training_output:
                if conditional:
                    if netG.Y == "label":
                        for l in range(netG.n_classes):
                            cond = torch.ones(64, dtype=int, device=device)*l
                            save_generator_images(epoch, netG, fixed_noise, model_path, conditional, cond)
                    elif netG.Y == "concentration":
                        for conc in concentrations:
                            conc = torch.ones((64,1), dtype=torch.float32, device=device)*conc
                            save_generator_images(epoch, netG, fixed_noise, model_path, conditional, conc)
                else:
                    save_generator_images(epoch, netG, fixed_noise, model_path, conditional)

            save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")
            
            netG.to(device)

    return netG

def WGAN_training(netG, netD, dataloader, optimizerG, optimizerD, model_path, conditional, n_classes, condition_sampling, 
                   epochs=cfg.hyperparameters.epochs, critic_iters=cfg.wgan.critic_iters, lambda_1=cfg.wgan.lambda_1,
                   lambda_2=cfg.wgan.lambda_2, save_generator_training_output=True, device=cfg.device):
    
    netG.train()
    netD.train()
    min_g_loss = np.inf

    ssim_module = SSIM(data_range=2, size_average=True, channel=1, nonnegative_ssim=True)

    concentrations = dataloader.dataset.df[netG.Y].unique().astype(np.float32)

    #wandb.watch((netG, netD), log="all", log_freq=len(dataloader))

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, netG.nz, device=device)

    # For each epoch
    gen_iters = 0
    for epoch in range(epochs):

        G_losses = []
        D_losses = []
        D_x_list = []
        gp_list = []
        ssim_list = []
        D_G_z1_list = []
        D_G_z2_list = []

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ############################
            if gen_iters < 10 or gen_iters % 500 == 0:
                critic_iterations = 50
            else:
                critic_iterations = critic_iters
            j=0
            while j < critic_iterations and i < len(dataloader):
                j += 1
                data = next(data_iter)
                i += 1
                # Format batch
                real_data = data[0].to(device)
                b_size, c, h, w = real_data.size()
                if conditional:
                    real_cond = data[1].to(device)
                    fake_cond = real_cond
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # for _ in range(critic_iters):
                # Generate batch of latent vectors
                noise = torch.randn(b_size, netG.nz, device=device)
                # Generate fake image batch with G
                fake_data = netG(noise, fake_cond) if conditional else netG(noise)
                # Forward pass real batch through D
                real_output = netD(real_data, real_cond) if conditional else netD(real_data)

                # Forward pass fake batch through D
                fake_output = netD(fake_data.detach(), fake_cond) if conditional else netD(fake_data.detach())

                # Generate interpolated data
                epsilon = torch.rand(b_size, 1, 1, 1, device=device)
                interpolated_data = epsilon*real_data + (1-epsilon)*fake_data
                # Forward pass interpolated batch through D
                interpolated_output = netD(interpolated_data, real_cond) if conditional else netD(interpolated_data)
                # Calculate gradients on interpolated data
                grad_x = torch.autograd.grad(outputs=interpolated_output,
                                                inputs=interpolated_data,
                                                grad_outputs=torch.ones(interpolated_output.size(), device=device, requires_grad=False),
                                                create_graph=True, retain_graph=True, only_inputs=True)
                grad_x = grad_x[0].view(b_size, -1)
                grad_x_norm = torch.sqrt(torch.sum(grad_x ** 2, dim=1))
                gp = torch.mean((grad_x_norm - 1.)**2)
                ssim_p = torch.mean(((torch.abs(real_data-fake_data)/ssim_module(real_data, fake_data)) - 1)**2)
                # WGAN-GP loss
                d_loss = torch.mean(fake_output) - torch.mean(real_output) + lambda_1*gp + lambda_2*ssim_p
                # Calculate gradients
                netD.zero_grad()
                d_loss.backward()
                # Update D
                optimizerD.step()

                D_losses.append(d_loss.item())
                gp_list.append(gp.item())
                ssim_list.append(ssim_p.item())
                D_x_list.append(real_output.mean().item())
                D_G_z1_list.append(fake_output.mean().item())

            ############################
            # (2) Update G network
            ###########################

            for p in netD.parameters():
                p.requires_grad = False # to avoid computation

            netG.zero_grad()
            # Generate batch of latent vectors
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise = torch.randn(dataloader.batch_size, cfg.model.nz, device=device)
            if condition_sampling == "balanced":
                rand_idx = torch.randint(len(concentrations), (dataloader.batch_size,))
                fake_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)
            else:
                if b_size != dataloader.batch_size:
                    rand_idx = torch.randint(len(concentrations), (dataloader.batch_size,))
                    fake_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)
            # Generate fake image batch with G
            fake_data = netG(noise, fake_cond) if conditional else netG(noise)
            # Forward pass fake batch through D
            fake_output_2 = netD(fake_data, fake_cond) if conditional else netD(fake_data)
            # Generator loss
            g_loss = -torch.mean(fake_output_2)
            # Calculate gradients for G
            g_loss.backward()
            # Update D
            optimizerG.step()

            gen_iters += 1

            G_losses.append(g_loss.item())
            D_G_z2_list.append(fake_output_2.mean().item())

            for p in netD.parameters():
                p.requires_grad = True
            
        G_loss = np.mean(G_losses)
        D_loss = np.mean(D_losses)
        Grad_penalty = np.mean(gp_list)
        SSIM_penalty = np.mean(ssim_list)
        D_x = np.mean(D_x_list)
        D_G_z1 = np.mean(D_G_z1_list)
        D_G_z2 = np.mean(D_G_z2_list)

        if G_loss < min_g_loss:
            min_g_loss = G_loss
            save_model(netG.cpu(), model_path, f"min_loss_{wandb.run.name}.pt")
            netG.to(device)

        logging_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "Grad_penalty": Grad_penalty,
            "SSIM_penalty": SSIM_penalty,
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2,
            }

        wandb.log(logging_dict)

        if epoch % 50 == 0:
            if save_generator_training_output:
                if conditional:
                    if netG.Y == "label":
                        for l in range(netG.n_classes):
                            cond = torch.ones(64, dtype=int, device=device)*l
                            save_generator_images(epoch, netG, fixed_noise, model_path, conditional, cond)
                    elif netG.Y == "concentration":
                        for conc in concentrations:
                            conc = torch.ones((64,1), dtype=torch.float32, device=device)*conc
                            save_generator_images(epoch, netG, fixed_noise, model_path, conditional, conc)
                else:
                    save_generator_images(epoch, netG, fixed_noise, model_path, conditional)

            save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")
            
            netG.to(device)

    return netG

def ACWGAN_training(netG, netD, dataloader, optimizerG, optimizerD, model_path, n_classes, condition_sampling,
                   epochs=cfg.hyperparameters.epochs, critic_iters=cfg.wgan.critic_iters, lambda_1=cfg.wgan.lambda_1,
                   lambda_2=cfg.wgan.lambda_2, save_generator_training_output=True, device=cfg.device):
    
    netG.train()
    netD.train()
    min_g_loss = np.inf

    ssim_module = SSIM(data_range=2, size_average=True, channel=1, nonnegative_ssim=True)

    concentrations = dataloader.dataset.df[netG.Y].unique().astype(np.float32)

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, netG.nz, device=device)

    aux_criterion = nn.CrossEntropyLoss() if netG.Y == "label" else nn.L1Loss()
    # For each epoch
    gen_iters = 0
    for epoch in range(epochs):
        # For each batch in the dataloader
        G_losses = []
        D_losses = []
        D_x_list = []
        gp_list = []
        ssim_list = []
        D_G_z1_list = []
        D_G_z2_list = []
        D_cond_losses = []
        G_z1_cond_losses = []
        G_z2_cond_losses = []

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            if gen_iters < 10 or gen_iters % 500 == 0:
                critic_iterations = 50
            else:
                critic_iterations = critic_iters
            j=0
            while j < critic_iterations and i < len(dataloader):
                j += 1
                data = next(data_iter)
                i += 1
                # Format batch
                real_data = data[0].to(device)
                b_size, c, h, w = real_data.size()
                real_cond = data[1].to(device)
                if condition_sampling == "real_frequencies":
                    fake_cond = real_cond
                elif condition_sampling == "balanced":
                    rand_idx = torch.randint(len(concentrations), (b_size,))
                    fake_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                #for _ in range(critic_iters):
                # Generate batch of latent vectors
                noise = torch.randn(b_size, netG.nz, device=device)
                # Generate fake image batch with G
                fake_data = netG(noise, fake_cond)

                # Forward pass real batch through D
                real_output, real_pred = netD(real_data)
                real_aux_loss = aux_criterion(real_pred, real_cond)

                # Forward pass fake batch through D
                fake_output, fake_pred = netD(fake_data.detach())
                fake_aux_loss = aux_criterion(fake_pred, fake_cond)

                # Generate interpolated data
                epsilon = torch.rand(b_size, 1, 1, 1, device=device)
                interpolated_data = epsilon*real_data + (1-epsilon)*fake_data
                if condition_sampling == "balanced":

                    interpolated_output, interpolated_pred = netD(interpolated_data)

                    grad_x = torch.autograd.grad(outputs=(interpolated_output, interpolated_pred),
                                                    inputs=interpolated_data,
                                                    grad_outputs=torch.ones(interpolated_output.size(), device=device, requires_grad=False),
                                                    create_graph=True, retain_graph=True, only_inputs=True)
                elif condition_sampling == "real_frequencies":
                    # Forward pass interpolated batch through D
                    interpolated_output, _ = netD(interpolated_data)
                    # Calculate gradients on interpolated data
                    grad_x = torch.autograd.grad(outputs=interpolated_output,
                                                    inputs=interpolated_data,
                                                    grad_outputs=torch.ones(interpolated_output.size(), device=device, requires_grad=False),
                                                    create_graph=True, retain_graph=True, only_inputs=True)
                grad_x = grad_x[0].view(b_size, -1)
                grad_x_norm = torch.sqrt(torch.sum(grad_x ** 2, dim=1))
                gp = torch.mean((grad_x_norm - 1.)**2)
                ssim_p = torch.mean(((torch.abs(real_data-fake_data)/ssim_module(real_data, fake_data)) - 1)**2)
                aux_loss = 0.5*(real_aux_loss + fake_aux_loss)
                # WGAN-GP loss
                d_loss = torch.mean(fake_output) - torch.mean(real_output) + aux_loss + lambda_1*gp + lambda_2*ssim_p
                # Calculate gradients
                netD.zero_grad()
                d_loss.backward()
                # Update D
                optimizerD.step()

                D_losses.append(d_loss.item())
                D_cond_losses.append(real_aux_loss.item())
                D_x_list.append(real_output.mean().item())
                gp_list.append(gp.item())
                ssim_list.append(ssim_p.item())
                D_G_z1_list.append(fake_output.mean().item())
                G_z1_cond_losses.append(fake_aux_loss.item())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # Generate batch of latent vectors
            noise = torch.randn(dataloader.batch_size, cfg.model.nz, device=device)
            if b_size != dataloader.batch_size:
                rand_idx = torch.randint(len(concentrations), (dataloader.batch_size,))
                fake_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)
            # Generate fake image batch with G
            fake_data = netG(noise, fake_cond)
            # Forward pass fake batch through D
            fake_output_2, fake_pred_2 = netD(fake_data)
            fake_aux_loss_2 = aux_criterion(fake_pred_2, fake_cond)
            # Generator loss
            g_loss = -torch.mean(fake_output_2) + fake_aux_loss_2
            # Calculate gradients for G
            g_loss.backward()
            # Update D
            optimizerG.step()

            gen_iters += 1

            G_losses.append(g_loss.item())
            D_G_z2_list.append(fake_output_2.mean().item())
            G_z2_cond_losses.append(fake_aux_loss_2.item())

            for p in netD.parameters():
                p.requires_grad = True

        G_loss = np.mean(G_losses)
        D_loss = np.mean(D_losses)
        D_x = np.mean(D_x_list)
        Grad_penalty = np.mean(gp_list)
        SSIM_penalty = np.mean(ssim_list)
        D_G_z1 = np.mean(D_G_z1_list)
        D_G_z2 = np.mean(D_G_z2_list)
        D_cond_loss = np.mean(D_cond_losses)
        G_z1_cond_loss = np.mean(G_z1_cond_losses)
        G_z2_cond_loss = np.mean(G_z2_cond_losses)

        if G_loss < min_g_loss:
            min_g_loss = G_loss
            save_model(netG.cpu(), model_path, f"min_loss_{wandb.run.name}.pt")
            netG.to(device)

        #ADD DISCRIMINATOR CLASSIFICATION METRICS
        logging_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "D_x": D_x,
            "Grad_penalty": Grad_penalty,
            "SSIM_penalty": SSIM_penalty,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2,
            "D_cond_loss": D_cond_loss,
            "G_z1_cond_loss": G_z1_cond_loss,
            "G_z2_cond_loss": G_z2_cond_loss
            }

        wandb.log(logging_dict)

        if epoch % 50 == 0:
            if save_generator_training_output:
                if netG.Y == "label":
                    for l in range(netG.n_classes):
                        cond = torch.ones(64, dtype=int, device=device)*l
                        save_generator_images(epoch, netG, fixed_noise, model_path, True, cond)
                elif netG.Y == "concentration":
                    for conc in concentrations:
                        conc = torch.ones((64,1), dtype=torch.float32, device=device)*conc
                        save_generator_images(epoch, netG, fixed_noise, model_path, True, conc)
            else:
                save_generator_images(epoch, netG, fixed_noise, model_path, True)

            save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")
            
            netG.to(device)

    return netG

def ACDCGAN_training(netG, netD, dataloader, criterion, optimizerG, optimizerD, model_path, n_classes, condition_sampling,
                        epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label, max_real_label=cfg.dcgan.max_real_label,
                        min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, p_noisy_label=cfg.dcgan.p_noisy_label,
                        save_generator_training_output=True, device=cfg.device):
    
    netG.train()
    netD.train()
    min_g_loss = np.inf

    concentrations = dataloader.dataset.df[netG.Y].unique().astype(np.float32)
    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, netG.nz, device=device) #noise_h, noise_w

    aux_criterion = nn.CrossEntropyLoss() if netG.Y == "label" else nn.L1Loss()
    # For each epoch
    gen_iters = 0
    for epoch in range(epochs):
        # For each batch in the dataloader
        G_losses = []
        D_losses = []
        D_x_list = []
        D_G_z1_list = []
        D_G_z2_list = []
        D_cond_losses = []
        G_z1_cond_losses = []
        G_z2_cond_losses = []

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ############################
            if gen_iters < 10 or gen_iters % 500 == 0:
                critic_iterations = 50
            else:
                critic_iterations = 5
            j=0
            while j < critic_iterations and i < len(dataloader):
                j += 1
                data = next(data_iter)
                i += 1
                # Format batch
                real_data = data[0].to(device)
                b_size, c, h, w = real_data.size()
                real_cond = data[1].to(device)
                if condition_sampling == "real_frequencies":
                    fake_cond = real_cond
                elif condition_sampling == "balanced":
                    fake_cond = torch.randint(0, n_classes, (b_size,), dtype=torch.long, device=device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                real_label = torch.zeros(b_size, device=device).uniform_(min_real_label, max_real_label)
                n_noisy = int(b_size*p_noisy_label)
                idx = torch.randint(b_size, (n_noisy,))
                try:
                    real_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_fake_label, max_fake_label)
                except:
                    pass
                real_label = real_label.unsqueeze(1)
                # Forward pass real batch through D
                real_output, real_pred = netD(real_data)
                # Calculate loss on all-real batch
                dis_errD_real = criterion(real_output, real_label)
                aux_errD_real = aux_criterion(real_pred, real_cond)
                errD_real = dis_errD_real + aux_errD_real
                D_x = nn.Sigmoid()(real_output).mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, netG.nz, device=device)
                # Generate fake image batch with G
                fake_data = netG(noise, fake_cond)
                fake_label = torch.zeros(b_size, device=device).uniform_(min_fake_label, max_fake_label)
                n_noisy = int(b_size*p_noisy_label)
                idx = torch.randint(b_size, (n_noisy,))
                try:
                    fake_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_real_label, max_real_label)
                except:
                    pass
                fake_label = fake_label.unsqueeze(1)
                # Classify all fake batch with D
                fake_output, fake_pred = netD(fake_data.detach())
                # Calculate D's loss on the all-fake batch
                dis_errD_fake = criterion(fake_output, fake_label)
                aux_errD_fake = aux_criterion(fake_pred, fake_cond)
                errD_fake = dis_errD_fake + aux_errD_fake

                errD = errD_real + errD_fake
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                netD.zero_grad()
                errD.backward()
                D_G_z1 = nn.Sigmoid()(fake_output).mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                D_losses.append(errD.item())
                D_x_list.append(real_output.mean().item())
                D_G_z1_list.append(fake_output.mean().item())
                D_cond_losses.append(aux_errD_real.item())
                G_z1_cond_losses.append(aux_errD_fake.item())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for p in netD.parameters():
                p.requires_grad = False
            # Since we just updated D, perform another forward pass of all-fake batch through D
            noise = torch.randn(dataloader.batch_size, cfg.model.nz, device=device)
            if b_size != dataloader.batch_size:
                rand_idx = torch.randint(len(concentrations), (dataloader.batch_size,))
                fake_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)
            fake_data = netG(noise, fake_cond)
            fake_output_2, fake_pred_2 = netD(fake_data)
            # Calculate G's loss based on this output
            real_label = torch.zeros(dataloader.batch_size, device=device).uniform_(min_real_label, max_real_label)
            n_noisy = int(dataloader.batch_size*p_noisy_label)
            idx = torch.randint(dataloader.batch_size, (n_noisy,))
            try:
                real_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_fake_label, max_fake_label)
            except:
                pass
            real_label = real_label.unsqueeze(1)
            dis_errG = criterion(fake_output_2, real_label) # fake labels are real for generator cost
            aux_errG = aux_criterion(fake_pred_2, fake_cond)
            errG = dis_errG + aux_errG
            # Calculate gradients for G
            netG.zero_grad()
            errG.backward()
            D_G_z2 = nn.Sigmoid()(fake_output_2).mean().item()
            # Update G
            optimizerG.step()

            gen_iters += 1

            G_losses.append(errG.item())
            D_G_z2_list.append(fake_output_2.mean().item())
            G_z2_cond_losses.append(aux_errG.item())

            for p in netD.parameters():
                p.requires_grad = True

        G_loss = np.mean(G_losses)
        D_loss = np.mean(D_losses)
        D_x = np.mean(D_x_list)
        D_G_z1 = np.mean(D_G_z1_list)
        D_G_z2 = np.mean(D_G_z2_list)
        D_cond_loss = np.mean(D_cond_losses)
        G_z1_cond_loss = np.mean(G_z1_cond_losses)
        G_z2_cond_loss = np.mean(G_z2_cond_losses)

        if G_loss < min_g_loss:
            min_g_loss = G_loss
            save_model(netG.cpu(), model_path, f"min_loss_{wandb.run.name}.pt")
            netG.to(device)

        #ADD DISCRIMINATOR CLASSIFICATION METRICS
        logging_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2,
            "D_cond_loss": D_cond_loss,
            "G_z1_cond_loss": G_z1_cond_loss,
            "G_z2_cond_loss": G_z2_cond_loss
            }

        wandb.log(logging_dict)

        if epoch % 50 == 0:
            if save_generator_training_output:
                if netG.Y == "label":
                    for l in range(netG.n_classes):
                        cond = torch.ones(64, dtype=int, device=device)*l
                        save_generator_images(epoch, netG, fixed_noise, model_path, True, cond)
                elif netG.Y == "concentration":
                    for conc in concentrations:
                        conc = torch.ones((64,1), dtype=torch.float32, device=device)*conc
                        save_generator_images(epoch, netG, fixed_noise, model_path, True, conc)

            save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")
            
            netG.to(device)

    return netG

def ACLSGAN_training(netG, netD, dataloader, criterion, optimizerG, optimizerD, model_path, n_classes, condition_sampling,
                        epochs=cfg.hyperparameters.epochs, min_real_label=cfg.dcgan.min_real_label, max_real_label=cfg.dcgan.max_real_label,
                        min_fake_label=cfg.dcgan.min_fake_label, max_fake_label=cfg.dcgan.max_fake_label, p_noisy_label=cfg.dcgan.p_noisy_label,
                        save_generator_training_output=True, device=cfg.device):
    
    netG.train()
    netD.train()
    min_g_loss = np.inf

    concentrations = dataloader.dataset.df[netG.Y].unique().astype(np.float32)
    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, netG.nz, device=device) #noise_h, noise_w

    aux_criterion = nn.CrossEntropyLoss() if netG.Y == "label" else nn.L1Loss()
    # For each epoch
    gen_iters = 0
    for epoch in range(epochs):
        # For each batch in the dataloader
        G_losses = []
        D_losses = []
        D_x_list = []
        D_G_z1_list = []
        D_G_z2_list = []
        D_cond_losses = []
        G_z1_cond_losses = []
        G_z2_cond_losses = []

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ############################
            if gen_iters < 10 or gen_iters % 500 == 0:
                critic_iterations = 50
            else:
                critic_iterations = 5
            j=0
            while j < critic_iterations and i < len(dataloader):
                j += 1
                data = next(data_iter)
                i += 1
                # Format batch
                real_data = data[0].to(device)
                b_size, c, h, w = real_data.size()
                real_cond = data[1].to(device)
                if condition_sampling == "real_frequencies":
                    fake_cond = real_cond
                elif condition_sampling == "balanced":
                    fake_cond = torch.randint(0, n_classes, (b_size,), dtype=torch.long, device=device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                real_label = torch.zeros(b_size, device=device).uniform_(min_real_label, max_real_label)
                n_noisy = int(b_size*p_noisy_label)
                idx = torch.randint(b_size, (n_noisy,))
                try:
                    real_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_fake_label, max_fake_label)
                except:
                    pass
                real_label = real_label.unsqueeze(1)
                # Forward pass real batch through D
                real_output, real_pred = netD(real_data)
                # Calculate loss on all-real batch
                dis_errD_real = 0.5 * torch.mean((real_output-real_label)**2)
                aux_errD_real = aux_criterion(real_pred, real_cond)
                errD_real = dis_errD_real + aux_errD_real

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, netG.nz, device=device)
                # Generate fake image batch with G
                fake_data = netG(noise, fake_cond)
                fake_label = torch.zeros(b_size, device=device).uniform_(min_fake_label, max_fake_label)
                n_noisy = int(b_size*p_noisy_label)
                idx = torch.randint(b_size, (n_noisy,))
                try:
                    fake_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_real_label, max_real_label)
                except:
                    pass
                fake_label = fake_label.unsqueeze(1)
                # Classify all fake batch with D
                fake_output, fake_pred = netD(fake_data.detach())
                # Calculate D's loss on the all-fake batch
                dis_errD_fake = 0.5 * torch.mean((fake_output-fake_label)**2)
                aux_errD_fake = aux_criterion(fake_pred, fake_cond)
                errD_fake = dis_errD_fake + aux_errD_fake

                errD = errD_real + errD_fake
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                netD.zero_grad()
                errD.backward()
                optimizerD.step()

                D_losses.append(errD.item())
                D_x_list.append(real_output.mean().item())
                D_G_z1_list.append(fake_output.mean().item())
                D_cond_losses.append(aux_errD_real.item())
                G_z1_cond_losses.append(aux_errD_fake.item())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for p in netD.parameters():
                p.requires_grad = False
            # Since we just updated D, perform another forward pass of all-fake batch through D
            noise = torch.randn(dataloader.batch_size, cfg.model.nz, device=device)
            if b_size != dataloader.batch_size:
                rand_idx = torch.randint(len(concentrations), (dataloader.batch_size,))
                fake_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)
            fake_data = netG(noise, fake_cond)
            fake_output_2, fake_pred_2 = netD(fake_data)
            # Calculate G's loss based on this output
            real_label = torch.zeros(dataloader.batch_size, device=device).uniform_(min_real_label, max_real_label)
            n_noisy = int(dataloader.batch_size*p_noisy_label)
            idx = torch.randint(dataloader.batch_size, (n_noisy,))
            try:
                real_label[idx] = torch.zeros(n_noisy, device=device).uniform_(min_fake_label, max_fake_label)
            except:
                pass
            real_label = real_label.unsqueeze(1)
            dis_errG = 0.5 * torch.mean((fake_output_2-real_label)**2) # fake labels are real for generator cost
            aux_errG = aux_criterion(fake_pred_2, fake_cond)
            errG = dis_errG + aux_errG
            # Calculate gradients for G
            netG.zero_grad()
            errG.backward()
            # Update G
            optimizerG.step()

            gen_iters += 1

            G_losses.append(errG.item())
            D_G_z2_list.append(fake_output_2.mean().item())
            G_z2_cond_losses.append(aux_errG.item())

            for p in netD.parameters():
                p.requires_grad = True

        G_loss = np.mean(G_losses)
        D_loss = np.mean(D_losses)
        D_x = np.mean(D_x_list)
        D_G_z1 = np.mean(D_G_z1_list)
        D_G_z2 = np.mean(D_G_z2_list)
        D_cond_loss = np.mean(D_cond_losses)
        G_z1_cond_loss = np.mean(G_z1_cond_losses)
        G_z2_cond_loss = np.mean(G_z2_cond_losses)

        if G_loss < min_g_loss:
            min_g_loss = G_loss
            save_model(netG.cpu(), model_path, f"min_loss_{wandb.run.name}.pt")
            netG.to(device)

        #ADD DISCRIMINATOR CLASSIFICATION METRICS
        logging_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2,
            "D_cond_loss": D_cond_loss,
            "G_z1_cond_loss": G_z1_cond_loss,
            "G_z2_cond_loss": G_z2_cond_loss
            }

        wandb.log(logging_dict)

        if epoch % 50 == 0:
            if save_generator_training_output:
                if netG.Y == "label":
                    for l in range(netG.n_classes):
                        cond = torch.ones(64, dtype=int, device=device)*l
                        save_generator_images(epoch, netG, fixed_noise, model_path, True, cond)
                elif netG.Y == "concentration":
                    for conc in concentrations:
                        conc = torch.ones((64,1), dtype=torch.float32, device=device)*conc
                        save_generator_images(epoch, netG, fixed_noise, model_path, True, conc)

            save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")
            
            netG.to(device)

    return netG

def VAE_training(netEnc, netDec, dataloader, optimizer, model_path, epochs, save_generator_training_output, device):
    netEnc.train()
    netDec.train()

    wandb.watch((netEnc, netDec), log="all", log_freq=len(dataloader))

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, netDec.nz, device=device)

    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        KLD_losses = []
        Recons_losses = []
        Losses = []
     
        for i, data in enumerate(dataloader, 0):

            # Format batch
            real_data = data[0].to(device)
            # Encode data
            mean, log_var = netEnc(real_data) #.view(b_size, -1)
            # mean = netEnc.mean(encoded_data)
            # log_var = netEnc.log_var(encoded_data)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mean) #.view(b_size, netDec.nz, 1, 1)
            # Decode data
            decoded_data = netDec(z)

            #Compute losses
            loss, recons_loss, kld_loss = compute_VAE_losses(decoded_data, real_data, mean, log_var)

            # Calculate gradients
            optimizer.zero_grad()
            loss.backward()
            # Update encoder and decoder
            optimizer.step()

            Losses.append(loss.item())
            Recons_losses.append(recons_loss.item())
            KLD_losses.append(kld_loss.item())
            
        if save_generator_training_output:
            save_generator_images(epoch, netDec, fixed_noise, model_path)
            save_decoded_images(epoch, netDec, z, model_path)

        loss = np.mean(Losses)
        recons_loss = np.mean(Recons_losses)
        kld_loss = np.mean(KLD_losses)

        logging_dict = {
            "Loss": loss,
            "Recons_loss": recons_loss,
            "KLD_loss": kld_loss
            }

        wandb.log(logging_dict)

    return netEnc, netDec

def starGAN_training(netG, netD, dataloader, optimizerG, optimizerD, model_path, epochs=cfg.hyperparameters.epochs,
                     critic_iters=cfg.wgan.critic_iters, lambda_1=cfg.wgan.lambda_1, lambda_cond=cfg.stargan.lambda_cond, 
                     lambda_recons=cfg.stargan.lambda_recons, lambda_gdl=cfg.stargan.lambda_gdl,
                     save_generator_training_output=True, device=cfg.device):
    
    netG.train()
    netD.train()
    min_g_loss = np.inf

    condition_loss = nn.CrossEntropyLoss() if netG.Y == "label" else nn.L1Loss()

    concentrations = dataloader.dataset.df[netG.Y].unique().astype(np.float32)

    # Create batch of fixed data that we will use to visualize
    # the progression of the generator
    fixed_imgs = torch.tensor([]).to(device)
    fixed_org_conds = torch.tensor([]).to(device)
    for data in dataloader:
        fixed_imgs = torch.concat([fixed_imgs, data[0].to(device)]) if len(fixed_imgs)>0 else data[0].to(device)
        fixed_org_conds = torch.concat([fixed_org_conds, data[1].to(device)]) if len(fixed_org_conds)>0 else data[1].to(device)
        if len(fixed_imgs)>=64:
            break
    fixed_imgs = fixed_imgs[:64]
    fixed_org_conds = fixed_org_conds[:64]
    rand_idx = torch.randint(len(concentrations), (64,))
    fixed_trg_conds = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)

    images_grid = vutils.make_grid(fixed_imgs, padding=2, normalize=True, value_range=(-1,1))
    images_path = model_path / "generated_images"
    images_path.mkdir(parents=True, exist_ok=True)
    img_path = images_path / "fixed_reference.png"
    vutils.save_image(images_grid, img_path)
    # For each epoch
    gen_iters = 0
    for epoch in range(epochs):
        # For each batch in the dataloader
        G_losses = []
        D_losses = []
        D_cond_losses = []
        G_cond_losses = []
        G_recons_losses = []
        gp_list = []
        gdl_list = []
        D_x_list = []
        D_G_z1_list = []
        D_G_z2_list = []

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ############################
            if gen_iters < 10 or gen_iters % 500 == 0:
                critic_iterations = 50
            else:
                critic_iterations = critic_iters
            j=0
            while j < critic_iterations and i < len(dataloader):
                j += 1
                data = next(data_iter)
                i += 1
                # Format batch
                real_img = data[0].to(device)
                org_cond = data[1].to(device)
                # org_cond = (org_cond / 400)

                b_size = org_cond.size(0)

                # Generate target domain labels randomly.
                rand_idx = torch.randint(len(concentrations), (b_size,))
                trg_cond = torch.tensor(concentrations)[rand_idx].unsqueeze(1).to(device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                #for _ in range(critic_iters):

                # Forward pass real batch through D
                real_output, real_cond_pred = netD(real_img)
                d_loss_cond = condition_loss(real_cond_pred, org_cond)

                # Forward pass fake batch through D
                fake_img = netG(real_img, trg_cond)
                fake_output, fake_cond_pred = netD(fake_img.detach())

                # Generate interpolated data
                epsilon = torch.rand(b_size, 1, 1, 1, device=device)
                interpolated_data = epsilon*real_img + (1-epsilon)*fake_img
                # Forward pass interpolated batch through D
                interpolated_output, _ = netD(interpolated_data)
                # Calculate gradients on interpolated data
                grad_x = torch.autograd.grad(outputs=interpolated_output,
                                                inputs=interpolated_data,
                                                grad_outputs=torch.ones(interpolated_output.size(), device=device, requires_grad=False),
                                                create_graph=True, retain_graph=True, only_inputs=True)
                grad_x = grad_x[0].view(b_size, -1)
                grad_x_norm = torch.sqrt(torch.sum(grad_x ** 2, dim=1))
                gp = torch.mean((grad_x_norm - 1.)**2)
                # WGAN-GP loss
                d_loss = torch.mean(fake_output) - torch.mean(real_output) + lambda_cond*d_loss_cond + lambda_1*gp
                # Calculate gradients
                netD.zero_grad()
                d_loss.backward()
                # Update D
                optimizerD.step()

                D_losses.append(d_loss.item())
                D_cond_losses.append(d_loss_cond.item())
                gp_list.append(gp.item())
                #gdl_list.append(gdl.item())
                D_x_list.append(real_output.mean().item())
                D_G_z1_list.append(fake_output.mean().item())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            # Generate fake image batch with G
            fake_img = netG(real_img, trg_cond)
            # Forward pass fake batch through D
            fake_output_2, fake_cond_pred = netD(fake_img)
            # Generator loss
            g_loss_cond = condition_loss(fake_cond_pred, trg_cond)
            # Target-to-original domain.
            img_reconst = netG(fake_img, org_cond)
            g_loss_rec = torch.mean(torch.abs(real_img - img_reconst))
            g_loss = -torch.mean(fake_output_2) + lambda_cond*g_loss_cond + lambda_recons*g_loss_rec
            # Calculate gradients for G
            netG.zero_grad()
            g_loss.backward()
            # Update D
            optimizerG.step()

            D_G_z2_list.append(fake_output_2.mean().item())
            G_losses.append(g_loss.item())
            G_cond_losses.append(g_loss_cond.item())
            G_recons_losses.append(g_loss_rec.item())

            for p in netD.parameters():
                p.requires_grad = True

        G_loss = np.mean(G_losses)
        D_loss = np.mean(D_losses)
        D_cond_loss = np.mean(D_cond_losses)
        G_cond_loss = np.mean(G_cond_losses)
        G_recons_loss = np.mean(G_recons_losses)
        Gradient_penalty = np.mean(gp_list)
        D_x = np.mean(D_x_list)
        D_G_z1 = np.mean(D_G_z1_list)
        D_G_z2 = np.mean(D_G_z2_list)

        if G_loss < min_g_loss:
            min_g_loss = G_loss
            save_model(netG.cpu(), model_path, f"min_loss_{wandb.run.name}.pt")
            netG.to(device)

        logging_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "G_cond_loss": G_cond_loss,
            "G_recons_loss": G_recons_loss,
            "D_cond_loss": D_cond_loss,
            "Gradient_penalty": Gradient_penalty,
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2,
            }

        wandb.log(logging_dict)

        if epoch % 50 == 0:
            if save_generator_training_output:
                if netG.Y == "label":
                    for l in range(netG.n_classes):
                        cond = torch.ones(64, dtype=int, device=device)*l
                        save_starGAN_images(netG, fixed_imgs, cond, epoch, model_path)
                elif netG.Y == "concentration":
                    for conc in concentrations:
                        cond = torch.ones((64, 1), dtype=torch.float32, device=device)*conc
                        save_starGAN_images(netG, fixed_imgs, cond, epoch, model_path)
                        
                    fake_img = netG(fixed_imgs, fixed_trg_conds)
                    img_reconst = netG(fake_img, fixed_org_conds).detach().cpu()
                    images_grid = vutils.make_grid(img_reconst, padding=2, normalize=True, value_range=(-1,1))
                    images_path = model_path / "generated_images"
                    img_path = images_path / f"epoch_{epoch}_reconstructed_reference.png"
                    vutils.save_image(images_grid, img_path)

            save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")
            
            netG.to(device)

    return netG

def PGGAN_training(cfg, netG, netD, optimizerG, optimizerD, model_path, conditional,
                   epochs=cfg.hyperparameters.epochs, schedule_start_epochs=cfg.pggan.start_epochs,  schedule_batch_size=cfg.pggan.batch_sizes, 
                   schedule_num_epochs=cfg.pggan.num_epochs, critic_iters=cfg.pggan.critic_iters, lambda_1=cfg.pggan.lambda_1, lambda_cond=cfg.pggan.lambda_cond,
                   save_generator_training_output=True, device=cfg.device):
    
    netG.train()
    netD.train()
    min_g_loss = np.inf

    condition_loss = nn.CrossEntropyLoss() if netG.Y == "label" else nn.L1Loss()

    batch_size=cfg.hyperparameters.batch_size
    dataloader = prepare_train_loader(split=cfg.data.split, augmented=cfg.data.augmented, height=4, width=9,
                                        min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                                        Y=cfg.model.Y, channels=cfg.data.channels, sobel=cfg.data.sobel, real_augmentation_factor=cfg.classifier.real_augmentation, 
                                        conc_intervals=cfg.data.concentration_intervals, normalize_conc=cfg.data.normalize_conc,
                                        batch_size=batch_size, apply_transforms=True)
    

    concentrations = dataloader.dataset.df[netG.Y].unique()

    #wandb.watch((netG, netD), log="all", log_freq=len(dataloader))

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(batch_size, netG.nz, 1, 1, device=device)

    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        G_losses = []
        D_losses = []
        D_cond_losses = []
        G_cond_losses = []
        D_x_list = []
        D_G_z1_list = []
        D_G_z2_list = []
	
        if epoch-1 in schedule_start_epochs:
            if (4*2**(netG.depth) <= 256) & (9*2**(netG.depth) <= 576):
                h, w = 4*2**(netG.depth), 9*2**(netG.depth)
                c = schedule_start_epochs.index(epoch-1)
                batch_size = schedule_batch_size[c]
                growing_epochs = schedule_num_epochs[c]
                dataloader = prepare_train_loader(split=cfg.data.split, augmented=cfg.data.augmented, height=h, width=w,
                                                    min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                                                    Y=cfg.model.Y, channels=cfg.data.channels, sobel=cfg.data.sobel, real_augmentation_factor=cfg.classifier.real_augmentation, 
                                                    conc_intervals=cfg.data.concentration_intervals, normalize_conc=cfg.data.normalize_conc,
                                                    batch_size=batch_size, apply_transforms=True)
                fixed_noise = fixed_noise[:batch_size]
                tot_iter_num = dataloader.dataset.__len__()/batch_size
                netG.growing_net(growing_epochs*tot_iter_num)
                netD.growing_net(growing_epochs*tot_iter_num)

                print("Output Resolution: %d x %d" % (h, w))
				
        for data in dataloader:
            # Format batch
            real_data = data[0].to(device)
            b_size, c, h, w = real_data.size()
            real_cond = data[1].to(device)
            fake_cond = real_cond
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            for p in netD.parameters():
                p.requires_grad = True
            for _ in range(critic_iters):
                # Generate batch of latent vectors
                noise = torch.randn(b_size, netG.nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake_data = netG(noise, fake_cond)
                # Forward pass real batch through D
                real_output, real_cond_pred = netD(real_data)
                d_loss_cond = condition_loss(real_cond_pred, real_cond)

                # Forward pass fake batch through D
                fake_output, _ = netD(fake_data.detach())

                # Generate interpolated data
                epsilon = torch.rand(b_size, 1, 1, 1, device=device)
                interpolated_data = epsilon*real_data + (1-epsilon)*fake_data
                # Forward pass interpolated batch through D
                interpolated_output, _ = netD(interpolated_data)
                # Calculate gradients on interpolated data
                grad_x = torch.autograd.grad(outputs=interpolated_output,
                                                inputs=interpolated_data,
                                                grad_outputs=torch.ones(interpolated_output.size(), device=device, requires_grad=False),
                                                create_graph=True, retain_graph=True, only_inputs=True)
                grad_x = grad_x[0].view(b_size, -1)
                grad_x_norm = torch.sqrt(torch.sum(grad_x ** 2, dim=1))

                # WGAN-GP loss
                d_loss = torch.mean(fake_output) - torch.mean(real_output) + lambda_cond*d_loss_cond + lambda_1*torch.mean((grad_x_norm - 1.)**2)
                # Calculate gradients
                netD.zero_grad()
                d_loss.backward()
                # Update D
                optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            # Generate batch of latent vectors
            noise = torch.randn(b_size, cfg.model.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake_data = netG(noise, fake_cond)
            # Forward pass fake batch through D
            fake_output_2, fake_cond_pred_2 = netD(fake_data)
            # Generator loss
            g_loss_cond = condition_loss(fake_cond_pred_2, fake_cond)
            g_loss = -torch.mean(fake_output_2) + lambda_cond*g_loss_cond
            # Calculate gradients for G
            netG.zero_grad()
            g_loss.backward()
            # Update D
            optimizerG.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            D_cond_losses.append(d_loss_cond.item())
            G_cond_losses.append(g_loss_cond.item())
            D_x_list.append(real_output.mean().item())
            D_G_z1_list.append(fake_output.mean().item())
            D_G_z2_list.append(fake_output_2.mean().item())

        G_loss = np.mean(G_losses)
        D_loss = np.mean(D_losses)
        D_cond_loss = np.mean(D_cond_losses)
        G_cond_loss = np.mean(G_cond_losses)
        D_x = np.mean(D_x_list)
        D_G_z1 = np.mean(D_G_z1_list)
        D_G_z2 = np.mean(D_G_z2_list)

        if G_loss < min_g_loss:
            min_g_loss = G_loss
            save_model(netG.cpu(), model_path, "min_loss_model.pt")
            netG.to(device)

        logging_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "D_cond_loss": D_cond_loss,
            "G_cond_loss": G_cond_loss,
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2,
            }

        wandb.log(logging_dict)

        if epoch % 5 == 0:
            if save_generator_training_output:
                if conditional:
                    if netG.Y == "label":
                        for l in range(netG.n_classes):
                            cond = torch.ones(batch_size, dtype=int, device=device)*l
                            save_generator_images(epoch, netG, fixed_noise, model_path, conditional, cond)
                    elif netG.Y == "concentration":
                        for conc in concentrations:
                            conc = torch.ones((batch_size,1), dtype=torch.float32, device=device)*conc
                            save_generator_images(epoch, netG, fixed_noise, model_path, conditional, conc)
                else:
                    save_generator_images(epoch, netG, fixed_noise, model_path, conditional)

            save_model(netG.cpu(), model_path, name=f"{wandb.run.name}.pt")
            
            netG.to(device)

    return netG

def compute_VAE_losses(recons_x, x, mu, log_var):
        
    """
    Computes the VAE loss function.
    """
    recons_loss = torch.sum(torch.nn.functional.mse_loss(recons_x.view(recons_x.shape[0],-1), x.view(x.shape[0],-1), reduction="none"), dim=1)
        
    kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)

    #w = min(1.5, (np.exp(epoch/(epochs/2))-1)/(np.e-1))
    #w = 0.75*(np.exp(epoch/(epochs/2))-1)/(np.e-1)
    #w = min(5, 5*(epoch/(epochs/2)))
    loss = torch.mean(recons_loss + kld_loss, dim=0)

    return loss, torch.mean(recons_loss.detach()), torch.mean(kld_loss.detach())

def gradient_img(x, device):
    a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, device=device)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device))
    G_x=conv1(x)

    b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, device=device)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).to(device))
    G_y=conv2(x)

    return G_x.abs(), G_y.abs()