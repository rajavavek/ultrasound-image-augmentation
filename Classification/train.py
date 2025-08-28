import torch
import torch.nn as nn
import numpy as np
from src.utils import set_seed, flatten_dict
from src.data import prepare_val_loader, prepare_train_loader, prepare_fake_loader, GANCollator, prepare_test_loader
from .model import get_classifier
from src.model import C_Generator, starGAN_Generator, save_model, load_model, weights_init
from src.PATHS import *
from src.config import cfg
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import copy
import wandb
import random
import hydra

@hydra.main(config_path=PROJECT_PATH.as_posix(), config_name="config", version_base=None)
def train_classifier(cfg):
    set_seed()
    if cfg.model.name == "starGAN":
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / cfg.model.name
        netG = starGAN_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), 
                                embedding_dim=cfg.model.embedding_dim, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, ndf=cfg.model.ndf, nc=cfg.model.nc)
        input_type = "img"
    else:
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / f"{cfg.model.name}_{cfg.model.gen_conditioning_mode}_{cfg.model.dis_conditioning_mode}" / (cfg.model.gen_name + "-" + cfg.model.dis_name)
        netG = C_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
            n_classes=0, conditioning_mode=cfg.model.gen_conditioning_mode, embedding_dim=cfg.model.embedding_dim,
            gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
        input_type = "noise"
    
    model_saved_name = cfg.model.save_name
    if cfg.classifier.use_real:
        if cfg.classifier.use_fake:
            saving_path = model_path / "classifier" / cfg.classifier.name

            netG = load_model(netG, model_path, model_saved_name)
            netG.to(cfg.device)
            netG.eval()

            if cfg.classifier.gan_augmentation <= 1:
                classifier_name = f"real_fake_classifier{'_truncated' if cfg.classifier.truncated else ''}_{model_saved_name}"
                train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                                                        min_conc=cfg.data.min_concentration,
                                                        max_conc=cfg.data.max_concentration, Y=cfg.classifier.Y,
                                                        channels=cfg.classifier.channels, sobel=cfg.classifier.sobel,
                                                        real_augmentation_factor=cfg.classifier.real_augmentation, 
                                                        conc_intervals=cfg.data.concentration_intervals,
                                                        batch_size=cfg.hyperparameters.batch_size, netG=netG, use_fake=True,
                                                        truncated=cfg.classifier.truncated, fake_concentrations=cfg.data.fake_concentrations, 
                                                        n_images_per_conc=cfg.classifier.n_images_per_conc_train)
                val_dataloader = prepare_val_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations, 
                                                    min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                                    Y=cfg.classifier.Y, channels=cfg.classifier.channels, sobel=cfg.classifier.sobel,
                                                    conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size,
                                                    netG=netG, use_fake=True, truncated=cfg.classifier.truncated,
                                                    fake_concentrations=cfg.data.fake_concentrations, n_images_per_conc=cfg.classifier.n_images_per_conc_val)

            else:
                classifier_name = f"{model_saved_name.split('.')[0]}_classifier_{cfg.classifier.nf}{f'_real_aug_{cfg.classifier.real_augmentation}' if cfg.classifier.real_augmentation > 1 else ''}{f'_gan_aug_{cfg.classifier.gan_augmentation}' if cfg.classifier.gan_augmentation > 1 else ''}.pt"
                train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                                                        min_conc=cfg.data.min_concentration,
                                                        max_conc=cfg.data.max_concentration, Y=cfg.classifier.Y,
                                                        channels=cfg.classifier.channels, sobel=cfg.classifier.sobel,
                                                        real_augmentation_factor=cfg.classifier.real_augmentation, 
                                                        conc_intervals=cfg.data.concentration_intervals,
                                                        batch_size=cfg.hyperparameters.batch_size, collate_fn=None)
                val_dataloader = prepare_val_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations, 
                                                    min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                                    Y=cfg.classifier.Y, channels=cfg.classifier.channels, sobel=cfg.classifier.sobel,
                                                    conc_intervals=cfg.data.concentration_intervals,
                                                    batch_size=cfg.hyperparameters.batch_size)
                conditions = np.sort(train_dataloader.dataset.df[cfg.model.Y].unique()).tolist()
                conditions = torch.tensor(list(set(conditions)))
                conditions, _ = torch.sort(conditions)

                _, weights = np.unique(train_dataloader.dataset.df[cfg.model.Y], return_counts=True)
                weights = torch.tensor(weights).float()

                n_to_generate = int(cfg.classifier.gan_augmentation*cfg.hyperparameters.batch_size) - cfg.hyperparameters.batch_size
                generator_batch_size = min(n_to_generate, cfg.classifier.gan_batch_size)

                collate_fn = GANCollator(netG, conditions, weights, channels=cfg.classifier.channels, sobel=not cfg.data.sobel,
                                        n_to_generate=n_to_generate, gan_sampling=cfg.classifier.gan_sampling, 
                                        generator_batch_size=generator_batch_size, input_type=input_type, transforms=None, device=cfg.device)
                
                train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                                                        min_conc=cfg.data.min_concentration,
                                                        max_conc=cfg.data.max_concentration, Y=cfg.classifier.Y,
                                                        channels=cfg.classifier.channels, sobel=cfg.classifier.sobel,
                                                        real_augmentation_factor=cfg.classifier.real_augmentation, 
                                                        conc_intervals=cfg.data.concentration_intervals,
                                                        batch_size=cfg.hyperparameters.batch_size, collate_fn=collate_fn)
        else:
            classifier_name = f"model_{cfg.classifier.nf}{f'_real_aug_{cfg.classifier.real_augmentation}' if cfg.classifier.real_augmentation > 1 else ''}{f'_gan_aug_{cfg.classifier.gan_augmentation}' if cfg.classifier.gan_augmentation > 1 else ''}.pt"
            saving_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.classifier.augmented else ''}{'_sobel' if cfg.classifier.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / "classifier" / cfg.classifier.name
            train_dataloader = prepare_train_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                                                    min_conc=cfg.data.min_concentration,
                                                    max_conc=cfg.data.max_concentration, Y=cfg.classifier.Y,
                                                    channels=cfg.classifier.channels, sobel=cfg.classifier.sobel,
                                                    real_augmentation_factor=cfg.classifier.real_augmentation, 
                                                    conc_intervals=cfg.data.concentration_intervals,
                                                    batch_size=cfg.hyperparameters.batch_size, augment_real=cfg.classifier.augment_real, 
                                                    n_images_per_conc=cfg.classifier.n_images_per_conc_train)
            val_dataloader = prepare_val_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations, 
                                                    min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                                    Y=cfg.classifier.Y, channels=cfg.classifier.channels, sobel=cfg.classifier.sobel,
                                                    conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size,
                                                    augment_real=cfg.classifier.augment_real, n_images_per_conc=cfg.classifier.n_images_per_conc_val)
    
    elif cfg.classifier.use_fake:
        classifier_name = f"fake_classifier{'_truncated' if cfg.classifier.truncated else ''}_{model_saved_name}"
        saving_path = model_path / "classifier" / cfg.classifier.name

        netG = load_model(netG, model_path, name=cfg.model.save_name)
        netG.to(cfg.device)
        netG.eval()

        conditions = cfg.data.fake_concentrations
        conditions = torch.tensor(list(set(conditions)))
        conditions, _ = torch.sort(conditions)
        n_conditions = len(conditions)

        weights = torch.ones(n_conditions, dtype=torch.float)
        n_images_train = cfg.classifier.n_images_per_conc_train * n_conditions
        n_images_val = cfg.classifier.n_images_per_conc_val * n_conditions

        collate_fn = GANCollator(netG, Y="label", conditions=conditions, weights=weights, channels=cfg.data.channels, sobel=not cfg.data.sobel, n_to_generate=0, gan_sampling="weighted",
                                 generator_batch_size=cfg.hyperparameters.batch_size, input_type=input_type, only_fake=True, truncated=cfg.classifier.truncated, transforms=None, device=cfg.device)

        train_dataloader = prepare_fake_loader(model_name=cfg.model.name, n_images=n_images_train, batch_size=cfg.hyperparameters.batch_size, collate_fn=collate_fn,
                                                Y=cfg.classifier.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                                sobel=cfg.data.sobel, channels=cfg.data.channels)
        
        val_dataloader = prepare_fake_loader(model_name=cfg.model.name, n_images=n_images_val, batch_size=cfg.hyperparameters.batch_size, collate_fn=collate_fn,
                                            Y=cfg.classifier.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                            sobel=cfg.data.sobel, channels=cfg.data.channels)

    try:
        train_concentrations = "_".join([str(float(conc)) for conc in np.unique(train_dataloader.dataset.df["concentration"])])
    except:
        train_concentrations = "_".join([str(float(conc)) for conc in conditions.tolist()])

    saving_path = saving_path / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) /train_concentrations
    saving_path.mkdir(parents=True, exist_ok=True)

    classifier = get_classifier(cfg.classifier.name, cfg.classifier.nf)
    classifier.to(cfg.device)
    classifier.apply(weights_init)

    optimizer = get_optimizer(classifier, lr=cfg.hyperparameters.lr,
                              beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2)

    try:
        config = {key: cfg._content[key] for key in ["data", "hyperparameters", "model", "classifier", "seed", "device"]}
    except:
        config = {key: value for key, value in cfg.items() if key in ["data", "hyperparameters", "model", "classifier", "seed", "device"]}
    
    wandb.init(project="classifier", entity="nbs-gan", config=flatten_dict(config))
    criterion = get_loss_function()
    real_fake = False
    classifier = classifier_training(classifier, real_fake, saving_path, classifier_name, train_dataloader, val_dataloader,
                                   criterion, optimizer, epochs=cfg.hyperparameters.epochs, device=cfg.device)


    save_model(classifier.cpu(), saving_path, classifier_name)
    
    wandb.finish()

def get_optimizer(model, lr=cfg.hyperparameters.lr, beta1=cfg.hyperparameters.beta1, beta2=cfg.hyperparameters.beta2):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    return optimizer

def get_loss_function(n_classes=len(cfg.data.concentration_intervals)):
    if n_classes > 1:
        return nn.CrossEntropyLoss()
    else:
        return nn.BCEWithLogitsLoss()

def train_one_augmented_epoch(model, real_fake, netG, dataloader, augmented_batch_size, criterion, optimizer, gan_n_classes=len(cfg.data.concentration_intervals),
                              channels=cfg.classifier.channels, sobel=cfg.classifier.sobel, gan_batch_size=cfg.classifier.gan_batch_size, device=cfg.device):
    
    model.train()

    epoch_loss = 0
    n_samples = 0
    pred_temp = []
    y_temp = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        gen_labels = []

        unique_labels, counts = labels.unique(return_counts=True)
        samples_per_label = int(augmented_batch_size / len(unique_labels))
        excess_samples = augmented_batch_size - (samples_per_label * len(unique_labels))
        extra_sample = random.sample(unique_labels.tolist(), excess_samples)

        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            n_to_generate = samples_per_label - count
            if label in extra_sample:
                n_to_generate += 1
            if n_to_generate > 0:
                gen_labels += [label]*n_to_generate
                weights = torch.zeros(gan_n_classes, dtype=torch.float)
                weights[label] = 1.0
                fake_loader = prepare_fake_loader(netG, weights, n_to_generate, conditional=True,
                                                channels=channels, sobel=sobel,
                                                batch_size=gan_batch_size, device=device)
                for data in fake_loader:
                    gen_images = data[0].to(cfg.device)
                    images = torch.concat([images, gen_images])
        
        if real_fake:
            real_labels = torch.ones_like(labels)
            fake_labels = torch.zeros_like(torch.tensor(gen_labels)).to(device)
            labels = torch.concat([real_labels, fake_labels])
        else:
            labels = torch.concat([labels, torch.tensor(gen_labels).to(device)])

        idx = torch.randperm(labels.shape[0])
        images = images[idx]
        labels = labels[idx]

        batch_size = images.size(0)
        n_samples += batch_size

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

        predicted_classes = torch.round(nn.Sigmoid()(outputs)) if real_fake else torch.max(outputs, 1)[1] # get class from network's prediction
        pred_temp+=predicted_classes.tolist()
        y_temp+=labels.to('cpu').tolist()

    prc=precision_score(y_temp, pred_temp, average='binary' if real_fake else 'macro', zero_division=0)
    rcl=recall_score(y_temp, pred_temp, average='binary' if real_fake else 'macro', zero_division=0)
    f1m=f1_score(y_temp, pred_temp, average='binary' if real_fake else 'macro')
    acr=accuracy_score(y_temp, pred_temp)
    epoch_loss /= n_samples

    torch.cuda.empty_cache()

    return epoch_loss, acr, prc, rcl, f1m

#Train functions
def train_one_epoch(model, real_fake, dataloader, criterion, optimizer, device):
    
    model.train()

    epoch_loss = 0
    n_samples = 0
    pred_temp = []
    y_temp = []

    for data in dataloader:
        images = data[0].to(device)
        if real_fake:
            labels = data[2].to(device)
        else:
            labels = data[1].to(device)
        
        batch_size = images.size(0)
        n_samples += batch_size

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

        predicted_classes = torch.round(nn.Sigmoid()(outputs)) if real_fake else torch.max(outputs, 1)[1] # get class from network's prediction
        pred_temp+=predicted_classes.tolist()
        y_temp+=labels.to('cpu').tolist()

    prc=precision_score(y_temp, pred_temp, average='binary' if real_fake else 'macro', zero_division=0)
    rcl=recall_score(y_temp, pred_temp, average='binary' if real_fake else 'macro', zero_division=0)
    f1m=f1_score(y_temp, pred_temp, average='binary' if real_fake else 'macro')
    acr=accuracy_score(y_temp, pred_temp)
    epoch_loss /= n_samples

    torch.cuda.empty_cache()

    return epoch_loss, acr, prc, rcl, f1m

@torch.no_grad()
def valid_one_epoch(model, real_fake, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0
    n_samples = 0
    pred_temp = []
    y_temp = []

    for data in dataloader:
        images = data[0].to(device)
        if real_fake:
            labels = data[2].to(device)
        else:
            labels = data[1].to(device)

        batch_size = images.size(0)
        n_samples += batch_size

        outputs = model(images)
        loss = criterion(outputs, labels)

        epoch_loss+=loss.item()

        predicted_classes = torch.round(nn.Sigmoid()(outputs)) if real_fake else torch.max(outputs, 1)[1] # get class from network's prediction
        pred_temp += predicted_classes.tolist()
        y_temp += labels.to('cpu').tolist()

    prc=precision_score(y_temp, pred_temp, average='binary' if real_fake else 'macro', zero_division=0)
    rcl=recall_score(y_temp, pred_temp, average='binary' if real_fake else 'macro', zero_division=0)
    f1m=f1_score(y_temp, pred_temp, average='binary' if real_fake else 'macro')
    acr=accuracy_score(y_temp, pred_temp)
    epoch_loss /= n_samples

    torch.cuda.empty_cache()

    return epoch_loss, acr, prc, rcl, f1m


def classifier_training(model, real_fake, saving_path, classifier_name, train_dataloader, validation_dataloader, criterion, optimizer,
                        epochs=cfg.hyperparameters.epochs, device=cfg.device):
    # To automatically log gradients
    #wandb.watch(model, log="all", log_freq=len(train_dataloader))

    best_f1m = 0
    for epoch in range(epochs):

        train_loss, train_acr, train_pcr, train_rcl, train_f1 = train_one_epoch(model, real_fake, train_dataloader, criterion, optimizer, device)

        val_loss, val_acr, val_pcr, val_rcl, val_f1 = valid_one_epoch(model, real_fake, validation_dataloader, criterion, device)

        logging_dict = {
        "train_loss": train_loss,
        "train_acr": train_acr,
        "train_pcr": train_pcr,
        "train_rcl": train_rcl,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_acr": val_acr,
        "val_pcr": val_pcr,
        "val_rcl": val_rcl,
        "val_f1": val_f1
        }

        wandb.log(logging_dict)
        if val_f1 > best_f1m:

            best_f1m = val_f1
            save_model(model.cpu(), saving_path, f"min_val_f1_{classifier_name}")
            model.to(device)

        save_model(model.cpu(), saving_path, classifier_name)
        model.to(device)

    return model