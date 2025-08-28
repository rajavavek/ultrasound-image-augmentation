import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from .datasets import InVitroDataset, FakeDataset, Real_Fake_Dataset
from src.PATHS import *
from src.config import cfg
from src.utils import set_seed

set_seed()

data_transforms = {
    "train": A.Compose([
        A.PadIfNeeded(min_height=cfg.data.height, min_width=cfg.data.width, p=1)
    ]),
    
    "val": A.Compose([]),

    "test": A.Compose([])
}

def prepare_train_loaders(dataset=cfg.data.dataset, split=cfg.data.split, augmented=cfg.data.augmented, train_concentrations=cfg.data.train_concentrations,
                          min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                          Y=cfg.model.Y, channels=cfg.data.channels, sobel=cfg.data.sobel, real_augmentation_factor=cfg.classifier.real_augmentation,
                          conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size,
                          train_shuffle=True, val_shuffle=False, collate_fn=None):

    train_loader = prepare_train_loader(dataset, split, augmented, train_concentrations, min_conc, max_conc, Y, channels, real_augmentation_factor, conc_intervals, batch_size, train_shuffle, collate_fn)
    val_loader = prepare_val_loader(dataset, split, augmented, train_concentrations, min_conc, max_conc, Y, channels, sobel, conc_intervals, batch_size, val_shuffle, collate_fn)
    
    return train_loader, val_loader

def prepare_train_loader(dataset=cfg.data.dataset, height=cfg.data.height, width=cfg.data.width,
                         train_concentrations=cfg.data.train_concentrations, min_conc=cfg.data.min_concentration, 
                         max_conc=cfg.data.max_concentration, Y=cfg.model.Y, channels=cfg.data.channels,
                         sobel=cfg.data.sobel, real_augmentation_factor=cfg.classifier.real_augmentation,
                         conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size, augment_real=False,
                         use_fake=False, truncated=cfg.regressor.truncated, netG=None, fake_concentrations=cfg.data.fake_concentrations, 
                         n_images_per_conc=cfg.regressor.n_images_per_conc_train, apply_transforms=False, shuffle=True, collate_fn=None):

    if dataset == "INVITRO_MENINGITIS":
        df = pd.read_csv(GAN_DATASET_PATH / "gan_invitro_data.csv")
        train_val_df = df[df["concentration"].isin(train_concentrations)]

        # Split by files
        group_train_val_df = train_val_df.groupby(["file number", "concentration"], as_index=False).count().sort_values(by=["file", "concentration"], ascending=[True, True])
        val_files = group_train_val_df.replace(5,9).drop_duplicates(subset="concentration")["file number"].values.tolist()
        train_files = list(set(group_train_val_df["file number"].values.tolist()) - set(val_files))
        train_df = df[df["file number"].isin(train_files)]

    elif dataset == "INVITRO_PERITONITIS":
        df = pd.read_csv(RAW_DATA_PATH / "list_peritonitis_regression_frames.csv", index_col=0)
        train_val_df = df[df["concentration"].isin(train_concentrations)]

        # Split by files
        group_train_val_df = train_val_df.groupby(["file number", "concentration"], as_index=False).count().sort_values(by=["concentration", "file"], ascending=[True, False])
        val_files = group_train_val_df.iloc[2::3, :]["file number"].values.tolist()
        train_files = list(set(group_train_val_df["file number"].values.tolist()) - set(val_files))
        train_df = df[df["file number"].isin(train_files)]

        train_df["file"] = RAW_DATA_PATH.as_posix() + "/FRAMES/" + train_df["file"]

    train_df = train_df[(train_df["concentration"]>=min_conc) & (train_df["concentration"]<=max_conc)]

    if augment_real:
        real_concentrations, real_concentrations_count = np.unique(train_df["concentration"].values, return_counts=True)
        for conc, count in zip(real_concentrations,real_concentrations_count):
            train_df = pd.concat([train_df, train_df[train_df["concentration"]==conc].sample(n_images_per_conc-count, replace=True, random_state=cfg.seed)])
    
    train_df.reset_index(drop=True, inplace=True)
    if use_fake:
        real_concentrations = np.unique(train_df["concentration"].values)
        fake_concentrations = np.array(list(set(fake_concentrations)))
        real_fake_concentrations, real_fake_concentrations_count = np.unique(np.concatenate([real_concentrations, fake_concentrations]), return_counts=True)
        final_concentrations = np.repeat(real_fake_concentrations, n_images_per_conc - real_fake_concentrations_count)
        final_concentrations = np.concatenate([final_concentrations, fake_concentrations])
        fake_df = pd.DataFrame(final_concentrations, columns=['concentration'])
        fake_df["file number"] = "noise"
        fake_df["file"] = "noise"
        train_df = pd.concat([train_df, fake_df], ignore_index=True)
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        if conc_intervals:
            train_df["label"] = train_df["concentration"].apply(get_labels, conc_intervals=conc_intervals)
            train_df = train_df[train_df["label"]<len(conc_intervals)]
        else:
            class_mapping = {conc:i for i, conc in enumerate(train_df["concentration"].unique())}
            train_df["label"] = train_df["concentration"].replace(class_mapping.keys(), class_mapping.values())

        train_dataset = Real_Fake_Dataset(train_df, netG, Y, channels, real_augmentation_factor, sobel, truncated=truncated,
                                          transforms=None)
    else:
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        if conc_intervals:
            train_df["label"] = train_df["concentration"].apply(get_labels, conc_intervals=conc_intervals)
            train_df = train_df[train_df["label"]<len(conc_intervals)]
        else:
            class_mapping = {conc:i for i, conc in enumerate(train_df["concentration"].unique())}
            train_df["label"] = train_df["concentration"].replace(class_mapping.keys(), class_mapping.values())

        train_dataset = InVitroDataset(train_df, Y, channels, real_augmentation_factor, sobel, transforms=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
    
    return train_loader

def prepare_val_loader(dataset=cfg.data.dataset, train_concentrations=cfg.data.train_concentrations,
                       min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                       Y=cfg.model.Y, channels=cfg.data.channels, sobel=cfg.data.sobel, conc_intervals=cfg.data.concentration_intervals,
                       batch_size=cfg.hyperparameters.batch_size, augment_real=False, use_fake=False, 
                       truncated=cfg.regressor.truncated, netG=None, fake_concentrations=cfg.data.fake_concentrations, 
                       n_images_per_conc=cfg.regressor.n_images_per_conc_val, shuffle=True, collate_fn=None, 
                       return_file=False, return_img_path=False):
    if dataset == "INVITRO_MENINGITIS":
        df = pd.read_csv(GAN_DATASET_PATH / "gan_invitro_data.csv")
        df = df[df["file"].str[-6:]=="NoFlip"]
        train_val_df = df[df["concentration"].isin(train_concentrations)]

        # Split by files
        group_train_val_df = train_val_df.groupby(["file number", "concentration"], as_index=False).count().sort_values(by=["file", "concentration"], ascending=[True, True])
        val_files = group_train_val_df.replace(5,9).drop_duplicates(subset="concentration")["file number"].values.tolist()
        val_df = df[df["file number"].isin(val_files)]
    
    elif dataset == "INVITRO_PERITONITIS":
        df = pd.read_csv(RAW_DATA_PATH / "list_peritonitis_regression_frames.csv", index_col=0)
        df = df[df["file"].str[-6:]=="NoFlip"]
        train_val_df = df[df["concentration"].isin(train_concentrations)]

        # Split by files
        group_train_val_df = train_val_df.groupby(["file number", "concentration"], as_index=False).count().sort_values(by=["concentration", "file"], ascending=[True, False])
        val_files = group_train_val_df.iloc[2::3, :]["file number"].values.tolist()
        val_df = df[df["file number"].isin(val_files)]
        
        val_df["file"] = RAW_DATA_PATH.as_posix() + "/FRAMES/" + val_df["file"]

    val_df = val_df[(val_df["concentration"]>=min_conc) & (val_df["concentration"]<=max_conc)]

    if augment_real:
        real_concentrations, real_concentrations_count = np.unique(val_df["concentration"].values, return_counts=True)
        for conc, count in zip(real_concentrations,real_concentrations_count):
            val_df = pd.concat([val_df, val_df[val_df["concentration"]==conc].sample(n_images_per_conc-count, replace=True, random_state=cfg.seed)])

    val_df.reset_index(drop=True, inplace=True)
    if use_fake:
        real_concentrations = np.unique(val_df["concentration"].values)
        fake_concentrations = np.array(list(set(fake_concentrations)))
        real_fake_concentrations, real_fake_concentrations_count = np.unique(np.concatenate([real_concentrations, fake_concentrations]), return_counts=True)
        final_concentrations = np.repeat(real_fake_concentrations, n_images_per_conc - real_fake_concentrations_count)
        final_concentrations = np.concatenate([final_concentrations, fake_concentrations])
        fake_df = pd.DataFrame(final_concentrations, columns=['concentration'])
        fake_df["file number"] = "noise"
        fake_df["file"] = "noise"
        val_df = pd.concat([val_df, fake_df], ignore_index=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)

        if conc_intervals:
            val_df["label"] = val_df["concentration"].apply(get_labels, conc_intervals=conc_intervals)
            val_df = val_df[val_df["label"]<len(conc_intervals)]
        else:
            class_mapping = {conc:i for i, conc in enumerate(val_df["concentration"].unique())}
            val_df["label"] = val_df["concentration"].replace(class_mapping.keys(), class_mapping.values())

        val_dataset = Real_Fake_Dataset(val_df, netG, Y, channels, sobel, truncated=truncated)
    else:
        val_df = val_df.sample(frac=1).reset_index(drop=True)
        if conc_intervals:
            val_df["label"] = val_df["concentration"].apply(get_labels, conc_intervals=conc_intervals)
            val_df = val_df[val_df["label"]<len(conc_intervals)]
        else:
            class_mapping = {conc:i for i, conc in enumerate(val_df["concentration"].unique())}
            val_df["label"] = val_df["concentration"].replace(class_mapping.keys(), class_mapping.values())
            
        val_dataset = InVitroDataset(val_df, Y, channels, sobel=sobel, transforms=None, return_file=return_file, return_img_path=return_img_path)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    return val_loader

def prepare_test_loader(dataset=cfg.data.dataset, split=cfg.data.split, augmented=False, test_concentrations=cfg.data.test_concentrations,
                        min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                        Y=cfg.model.Y, channels=cfg.data.channels, sobel=cfg.data.sobel,
                        conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size,shuffle=True,
                        collate_fn=None, return_file=False, return_img_path=False):
    if dataset == "INVITRO_MENINGITIS":
        df = pd.read_csv(GAN_DATASET_PATH / "gan_invitro_data.csv")
        df = df[df["file"].str[-6:]=="NoFlip"]
        df.reset_index(drop=True, inplace=True)
        test_df = df[df["concentration"].isin(test_concentrations)]
    elif dataset == "INVITRO_PERITONITIS":
        df = pd.read_csv(RAW_DATA_PATH / "list_peritonitis_regression_frames.csv", index_col=0)
        df = df[df["file"].str[-6:]=="NoFlip"]
        df.reset_index(drop=True, inplace=True)
        test_df = df[df["concentration"].isin(test_concentrations)]
        test_df["file"] = RAW_DATA_PATH.as_posix() + "/FRAMES/" + test_df["file"]

    test_df = test_df[(test_df["concentration"]>=min_conc) & (test_df["concentration"]<=max_conc)]
    if conc_intervals:
        test_df["label"] = test_df["concentration"].apply(get_labels, conc_intervals=conc_intervals)
        test_df = test_df[test_df["label"]<len(conc_intervals)]
    else:
        class_mapping = {conc:i for i, conc in enumerate(test_df["concentration"].unique())}
        test_df["label"] = test_df["concentration"].replace(class_mapping.keys(), class_mapping.values())

    test_df.reset_index(drop=True, inplace=True)
    test_dataset = InVitroDataset(test_df, Y, channels, sobel=sobel, return_file=return_file, return_img_path=return_img_path,
                                  transforms=None)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    return test_loader

def prepare_fake_loader(model_name, n_images, batch_size, collate_fn, Y=cfg.model.Y, 
                        min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                        sobel=cfg.data.sobel, channels=cfg.data.channels):
    if model_name == "starGAN":
        train_df = pd.read_csv(GAN_DATASET_PATH / f"gan_invitro_data_train_by_conc.csv")
        train_df = train_df[(train_df["concentration"]>=min_conc) & (train_df["concentration"]<=max_conc)]
        if len(train_df) >= n_images:
            replace=False
        else:
            replace=True
        train_df = train_df.sample(n=n_images, replace=replace, random_state=cfg.seed)

        train_df.reset_index(drop=True, inplace=True)
        train_dataset = InVitroDataset(train_df, Y, channels=channels, sobel=sobel, transforms=None)

        fake_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=False, collate_fn=collate_fn)
    else:
        fake_dataset = FakeDataset(n_images)
        fake_loader = DataLoader(fake_dataset, batch_size=batch_size, drop_last=False, shuffle=False, collate_fn=collate_fn)

    return fake_loader

def get_labels(img_conc, conc_intervals):
    y=0
    for conc in conc_intervals:
        lower_conc, higher_conc = conc.split("-")
        if higher_conc == "inf":
            higher_conc = np.inf
        else:
            higher_conc = int(higher_conc)
        if (img_conc >= int(lower_conc)) & (img_conc <= higher_conc):
            return y
        else:
            y+=1
    return y