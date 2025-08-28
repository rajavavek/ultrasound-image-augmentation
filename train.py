import torch
import torch.nn as nn
import numpy as np
from src.utils import set_seed, flatten_dict
from src.data import prepare_val_loader, prepare_train_loader, prepare_fake_loader, GANCollator
from .model import get_regressor
from src.model import C_Generator, starGAN_Generator, save_model, load_model, weights_init
from src.PATHS import *
from src.config import cfg
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import copy
import wandb
import random
import hydra

@hydra.main(config_path=PROJECT_PATH.as_posix(), config_name="config", version_base=None)
def train_regressor(cfg):

    set_seed()
    if cfg.model.name == "starGAN":
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / cfg.model.name
        netG = starGAN_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), 
                                embedding_dim=cfg.model.embedding_dim, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, ndf=cfg.model.ndf, nc=cfg.model.nc)
        input_type = "img"
    else:
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / f"{cfg.model.name}_{cfg.model.gen_conditioning_mode}_{cfg.model.dis_conditioning_mode}" / (cfg.model.gen_name + "-" + cfg.model.dis_name)
        netG = C_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
            n_classes=0, conditioning_mode=cfg.model.gen_conditioning_mode, embedding_dim=cfg.model.embedding_dim,
            gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
        input_type = "noise"
    
    model_saved_name = cfg.model.save_name
    if cfg.regressor.use_real:
        if cfg.regressor.use_fake:
            saving_path = model_path / "regressor" / cfg.regressor.name

            netG = load_model(netG, model_path, model_saved_name)
            netG.to(cfg.device)
            netG.eval()

            if cfg.regressor.gan_augmentation <= 1:
                regressor_name = f"real_fake_regressor{'_truncated' if cfg.regressor.truncated else ''}_{model_saved_name}"
                train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                                                        min_conc=cfg.data.min_concentration,
                                                        max_conc=cfg.data.max_concentration, Y=cfg.regressor.Y,
                                                        channels=cfg.regressor.channels, sobel=cfg.regressor.sobel,
                                                        real_augmentation_factor=cfg.regressor.real_augmentation, 
                                                        conc_intervals=cfg.data.concentration_intervals,
                                                        batch_size=cfg.hyperparameters.batch_size, netG=netG, use_fake=True,
                                                        truncated=cfg.regressor.truncated, fake_concentrations=cfg.data.fake_concentrations, 
                                                        n_images_per_conc=cfg.regressor.n_images_per_conc_train)
                val_dataloader = prepare_val_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations, 
                                                    min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                                    Y=cfg.regressor.Y, channels=cfg.regressor.channels, sobel=cfg.regressor.sobel,
                                                    conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size,
                                                    netG=netG, use_fake=True, truncated=cfg.regressor.truncated,
                                                    fake_concentrations=cfg.data.fake_concentrations, n_images_per_conc=cfg.regressor.n_images_per_conc_val)

            else:
                regressor_name = f"{model_saved_name.split('.')[0]}_regressor_{cfg.regressor.nf}{f'_real_aug_{cfg.regressor.real_augmentation}' if cfg.regressor.real_augmentation > 1 else ''}{f'_gan_aug_{cfg.regressor.gan_augmentation}' if cfg.regressor.gan_augmentation > 1 else ''}.pt"
                train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                                                        min_conc=cfg.data.min_concentration,
                                                        max_conc=cfg.data.max_concentration, Y=cfg.regressor.Y,
                                                        channels=cfg.regressor.channels, sobel=cfg.regressor.sobel,
                                                        real_augmentation_factor=cfg.regressor.real_augmentation, 
                                                        conc_intervals=cfg.data.concentration_intervals,
                                                        batch_size=cfg.hyperparameters.batch_size, collate_fn=None)
                val_dataloader = prepare_val_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations, 
                                                    min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                                    Y=cfg.regressor.Y, channels=cfg.regressor.channels, sobel=cfg.regressor.sobel,
                                                    conc_intervals=cfg.data.concentration_intervals,
                                                    batch_size=cfg.hyperparameters.batch_size)
                conditions = np.sort(train_dataloader.dataset.df[cfg.model.Y].unique()).tolist()
                conditions = torch.tensor(list(set(conditions)))
                conditions, _ = torch.sort(conditions)

                _, weights = np.unique(train_dataloader.dataset.df[cfg.model.Y], return_counts=True)
                weights = torch.tensor(weights).float()

                n_to_generate = int(cfg.regressor.gan_augmentation*cfg.hyperparameters.batch_size) - cfg.hyperparameters.batch_size
                generator_batch_size = min(n_to_generate, cfg.regressor.gan_batch_size)

                collate_fn = GANCollator(netG, conditions, weights, channels=cfg.regressor.channels, sobel=not cfg.data.sobel,
                                        n_to_generate=n_to_generate, gan_sampling=cfg.regressor.gan_sampling, 
                                        generator_batch_size=generator_batch_size, input_type=input_type, transforms=None, device=cfg.device)
                
                train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                                                        min_conc=cfg.data.min_concentration,
                                                        max_conc=cfg.data.max_concentration, Y=cfg.regressor.Y,
                                                        channels=cfg.regressor.channels, sobel=cfg.regressor.sobel,
                                                        real_augmentation_factor=cfg.regressor.real_augmentation, 
                                                        conc_intervals=cfg.data.concentration_intervals,
                                                        batch_size=cfg.hyperparameters.batch_size, collate_fn=collate_fn)
        else:
            regressor_name = f"model_{cfg.regressor.nf}{f'_real_aug_{cfg.regressor.real_augmentation}' if cfg.regressor.real_augmentation > 1 else ''}{f'_gan_aug_{cfg.regressor.gan_augmentation}' if cfg.regressor.gan_augmentation > 1 else ''}.pt"
            saving_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.regressor.augmented else ''}{'_sobel' if cfg.regressor.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / "regressor" / cfg.regressor.name
            train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                                                    min_conc=cfg.data.min_concentration,
                                                    max_conc=cfg.data.max_concentration, Y=cfg.regressor.Y,
                                                    channels=cfg.regressor.channels, sobel=cfg.regressor.sobel,
                                                    real_augmentation_factor=cfg.regressor.real_augmentation, 
                                                    conc_intervals=cfg.data.concentration_intervals,
                                                    batch_size=cfg.hyperparameters.batch_size, augment_real=cfg.regressor.augment_real, 
                                                    n_images_per_conc=cfg.regressor.n_images_per_conc_train)
            val_dataloader = prepare_val_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations, 
                                                    min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                                    Y=cfg.regressor.Y, channels=cfg.regressor.channels, sobel=cfg.regressor.sobel,
                                                    conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size,
                                                    augment_real=cfg.regressor.augment_real, n_images_per_conc=cfg.regressor.n_images_per_conc_val)
    
    elif cfg.regressor.use_fake:
        regressor_name = f"fake_regressor{'_truncated' if cfg.regressor.truncated else ''}_{model_saved_name}"
        saving_path = model_path / "regressor" / cfg.regressor.name

        netG = load_model(netG, model_path, name=cfg.model.save_name)
        netG.to(cfg.device)
        netG.eval()

        conditions = cfg.data.fake_concentrations
        conditions = torch.tensor(list(set(conditions)))
        conditions, _ = torch.sort(conditions)
        n_conditions = len(conditions)

        weights = torch.ones(n_conditions, dtype=torch.float)
        n_images_train = cfg.regressor.n_images_per_conc_train * n_conditions
        n_images_val = cfg.regressor.n_images_per_conc_val * n_conditions

        collate_fn = GANCollator(netG, conditions=conditions, weights=weights, channels=cfg.data.channels, sobel=not cfg.data.sobel, n_to_generate=0, gan_sampling="weighted",
                                generator_batch_size=cfg.hyperparameters.batch_size, input_type=input_type, only_fake=True, truncated=cfg.regressor.truncated, transforms=None, device=cfg.device)

        train_dataloader = prepare_fake_loader(model_name=cfg.model.name, n_images=n_images_train, batch_size=cfg.hyperparameters.batch_size, collate_fn=collate_fn,
                                                Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                                sobel=cfg.data.sobel, channels=cfg.data.channels)
        
        val_dataloader = prepare_fake_loader(model_name=cfg.model.name, n_images=n_images_val, batch_size=cfg.hyperparameters.batch_size, collate_fn=collate_fn,
                                            Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                            sobel=cfg.data.sobel, channels=cfg.data.channels)

    try:
        train_concentrations = "_".join([str(float(conc)) for conc in np.unique(train_dataloader.dataset.df["concentration"])])
    except:
        train_concentrations = "_".join([str(float(conc)) for conc in conditions.tolist()])

    saving_path = saving_path / train_concentrations
    saving_path.mkdir(parents=True, exist_ok=True)

    regressor = get_regressor(cfg.regressor.name, cfg.regressor.nf)
    regressor.to(cfg.device)
    regressor.apply(weights_init)

    optimizer = get_optimizer(regressor, lr=cfg.hyperparameters.lr,
                              beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2)

    try:
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "regressor", "seed", "device"]}
    except:
        config = {key: value for key, value in cfg.items() if key in ["data", "hyperparameters", "model", "regressor", "seed", "device"]}
    
    wandb.init(project="regressor", entity="nbs-gan", config=flatten_dict(config))

    criterion = get_loss_function(cfg.regressor.loss)
    regressor = regressor_training(regressor, saving_path, regressor_name, train_dataloader, val_dataloader,
                                   criterion, optimizer, epochs=cfg.hyperparameters.epochs, device=cfg.device)

    save_model(regressor.cpu(), saving_path, regressor_name)
    #wandb.finish()

def get_optimizer(model, lr=cfg.hyperparameters.lr, beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizer

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        pred[pred<0]=0
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))
    
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        pred[pred<0]=0
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    
def get_loss_function(loss_name=cfg.regressor.loss):
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "msle":
        return MSLELoss()
    elif loss_name == "rmsle":
        return RMSLELoss()

#Train functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    
    model.train()

    epoch_loss = 0
    n_samples = 0
    pred_temp = []
    y_temp = []

    for data in dataloader:
        images = data[0].to(device)
        concentrations = data[1].to(device)

        batch_size = images.size(0)
        n_samples += batch_size

        pred_concentrations = model(images)
        loss = criterion(pred_concentrations, concentrations)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

        pred_temp += pred_concentrations.tolist()
        y_temp += concentrations.tolist()

    mse = mean_squared_error(y_temp, pred_temp)
    rmse = mean_squared_error(y_temp, pred_temp, squared=False)
    mae = mean_absolute_error(y_temp, pred_temp)
    y_temp = np.array(y_temp)
    y_temp[y_temp == 0] = 1.0
    mape = mean_absolute_percentage_error(list(y_temp), pred_temp)

    epoch_loss /= n_samples

    torch.cuda.empty_cache()

    return epoch_loss, mse, rmse, mae, mape

@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0
    n_samples = 0
    pred_temp = []
    y_temp = []

    for data in dataloader:
        images = data[0].to(device)
        concentrations = data[1].to(device)

        batch_size = images.size(0)
        n_samples += batch_size

        pred_concentrations = model(images)
        loss = criterion(pred_concentrations, concentrations)

        epoch_loss+=loss.item()

        pred_temp += pred_concentrations.tolist()
        y_temp += concentrations.tolist()

    mse = mean_squared_error(y_temp, pred_temp)
    rmse = mean_squared_error(y_temp, pred_temp, squared=False)
    mae = mean_absolute_error(y_temp, pred_temp)
    y_temp = np.array(y_temp)
    y_temp[y_temp == 0] = 1.0
    mape = mean_absolute_percentage_error(list(y_temp), pred_temp)

    epoch_loss /= n_samples

    torch.cuda.empty_cache()

    return epoch_loss, mse, rmse, mae, mape


def regressor_training(model, saving_path, regressor_name, train_dataloader, validation_dataloader, criterion, optimizer,
                       epochs=cfg.hyperparameters.epochs, device=cfg.device):
    # To automatically log gradients
    #wandb.watch(model, log="all", log_freq=len(train_dataloader))

    best_loss = np.inf
    for epoch in range(epochs):

        train_loss, train_mse, train_rmse, train_mae, train_mape = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_mse, val_rmse, val_mae, val_mape = valid_one_epoch(model, validation_dataloader, criterion, device)

        logging_dict = {
        "train_loss": train_loss,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_mape": train_mape,
        "val_loss": val_loss,
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_mape": val_mape
        }

        wandb.log(logging_dict)
        # deep copy the model weights
        if val_loss < best_loss:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = val_loss
            save_model(model.cpu(), saving_path, f"min_val_loss_{regressor_name}")
            model.to(device)

    # load best model weights
    # model.load_state_dict(best_model_wts)
        save_model(model.cpu(), saving_path, regressor_name)
        model.to(device)
    return model