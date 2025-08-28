import pandas as pd
import numpy as np
import torch
from src.PATHS import MODELS_PATH, PROJECT_PATH, MODELS_METRICS_CSV
from src.data import prepare_train_loader, prepare_val_loader, prepare_test_loader, prepare_fake_loader, GANCollator
from src.model.conditional_model import C_Generator
from src.model.model import Generator
from src.model.utils import load_model
from src.model.starGAN import starGAN_Generator
from src.utils import set_seed
from .is_fid_metrics import ConditionalFIDandIS, FIDandIS
from tqdm import tqdm
import hydra

#TODO: Adapt for Conditional WGAN - Option to imitate frequency from original dataset
#numpy.random.choice(list_of_candidates, number_of_items_to_pick, p=probability_distribution, replace=True)
# or random.choices(population, weights, k)
@hydra.main(config_path=PROJECT_PATH.as_posix(), config_name="config", version_base=None)
def compute_is_and_fid(cfg):

    set_seed()
    model_save_name = cfg.model.save_name[:-3]
    train_loader = prepare_train_loader(dataset=cfg.data.dataset, height=cfg.data.height, width=cfg.data.width,
                                    train_concentrations=cfg.data.train_concentrations, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                                    Y=cfg.model.Y, channels=3, sobel=cfg.data.sobel, real_augmentation_factor=cfg.classifier.real_augmentation, 
                                    conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size, apply_transforms=True)
    val_loader = prepare_val_loader(dataset=cfg.data.dataset,train_concentrations=cfg.data.train_concentrations,
                                    min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                                    Y=cfg.model.Y, channels=3, sobel=cfg.data.sobel, 
                                    conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size)
    test_loader = prepare_test_loader(dataset=cfg.data.dataset, test_concentrations=cfg.data.test_concentrations,
                                     min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration,
                                     Y=cfg.model.Y, channels=3, sobel=cfg.data.sobel,
                                     conc_intervals=cfg.data.concentration_intervals, batch_size=cfg.hyperparameters.batch_size)
    conditions = []
    if cfg.evaluation.use_val:
        conditions += np.sort(val_loader.dataset.df[cfg.model.Y].unique()).tolist()
    if cfg.evaluation.use_test:
        conditions += np.sort(test_loader.dataset.df[cfg.model.Y].unique()).tolist()
    conditions = torch.tensor(list(set(conditions)))
    conditions, _ = torch.sort(conditions)
    index_mapping = {cond: idx for idx, cond in enumerate(conditions.int().tolist())}
    n_conditions = len(conditions)

    train_conditions = np.sort(train_loader.dataset.df[cfg.model.Y].unique()).tolist()
    train_conditions = torch.tensor(list(set(train_conditions)))
    train_conditions, _ = torch.sort(train_conditions)
    train_index_mapping = {cond: idx for idx, cond in enumerate(train_conditions.int().tolist())}
    n_train_conditions = len(train_conditions)

    train_test_conditions = []
    if cfg.evaluation.use_val:
        train_test_conditions += np.sort(val_loader.dataset.df[cfg.model.Y].unique()).tolist()
    if cfg.evaluation.use_test:
        train_test_conditions += np.sort(test_loader.dataset.df[cfg.model.Y].unique()).tolist()
    train_test_conditions += np.sort(train_loader.dataset.df[cfg.model.Y].unique()).tolist()
    train_test_conditions = torch.tensor(list(set(train_test_conditions)))
    train_test_conditions, _ = torch.sort(train_test_conditions)
    train_test_index_mapping = {cond: idx for idx, cond in enumerate(train_test_conditions.int().tolist())}
    n_train_test_conditions = len(train_test_conditions)


    model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / f"{cfg.model.name}_{cfg.model.gen_conditioning_mode}_{cfg.model.dis_conditioning_mode}" / (cfg.model.gen_name + "-" + cfg.model.dis_name)
    if cfg.model.name == "starGAN":
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / cfg.model.name
        netG = starGAN_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, n_classes=len(cfg.data.concentration_intervals), 
                                 embedding_dim=cfg.model.embedding_dim, gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, ndf=cfg.model.ndf, nc=cfg.model.nc)
        input_type = "img"
        C_FID_IS = ConditionalFIDandIS(n_conditions=n_conditions).to(cfg.device)
        conditional = True
    elif (cfg.model.name.startswith("C-") or cfg.model.name.startswith("AC-")):
        netG = C_Generator(Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                            n_classes=0, conditioning_mode=cfg.model.gen_conditioning_mode, embedding_dim=cfg.model.embedding_dim,
                            gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
        input_type = "noise"
        C_FID_IS = ConditionalFIDandIS(n_conditions=n_conditions).to(cfg.device)
        conditional = True
    else:
        netG = Generator(gen_name=cfg.model.gen_name, nz=cfg.model.nz, ngf=cfg.model.ngf, nc=cfg.model.nc)
        input_type = "noise"
        conditional = False

    FID_IS = FIDandIS().to(cfg.device)
    
    netG = load_model(netG, model_path, name=cfg.model.save_name)
    netG.to(cfg.device)
    netG.eval()

    print("Real_val_test-fake evaluation")
    if cfg.evaluation.use_val:
        _, weights = np.unique(val_loader.dataset.df[cfg.model.Y], return_counts=True)
        for data in tqdm(val_loader, desc="Loading and computing features from validation set"):
            imgs = data[0].to(cfg.device)
            if conditional:
                batch_conds = data[1].int().view(-1).tolist()
                cond_idxs = [index_mapping.get(item) for item in batch_conds]
                C_FID_IS.update(imgs, cond_idxs, real=True)
            FID_IS.update(imgs, real=True)
            
    if cfg.evaluation.use_test:
        _, weights = np.unique(test_loader.dataset.df[cfg.model.Y], return_counts=True)
        for data in tqdm(test_loader, desc="Loading and computing features from test set"):
            imgs = data[0].to(cfg.device)
            if conditional:
                batch_conds = data[1].int().view(-1).tolist()
                cond_idxs = [index_mapping.get(item) for item in batch_conds]
                C_FID_IS.update(imgs, cond_idxs, real=True)
            FID_IS.update(imgs, real=True)
    if cfg.evaluation.use_val and cfg.evaluation.use_test:
        _, weights = np.unique(pd.concat([val_loader.dataset.df[cfg.model.Y], test_loader.dataset.df[cfg.model.Y]]), return_counts=True)
            
    if cfg.evaluation.sampling == "real_frequencies":
        weights = torch.tensor(weights, dtype=torch.float)
    elif cfg.evaluation.sampling == "balanced":
        weights = torch.ones(n_conditions, dtype=torch.float)

    collate_fn = GANCollator(netG, conditions=conditions, weights=weights, channels=3, sobel=not cfg.data.sobel, n_to_generate=cfg.evaluation.batch_size, gan_sampling="weighted",
                             generator_batch_size=cfg.evaluation.batch_size, input_type=input_type, only_fake=True, transforms=None, device=cfg.device)

    fake_loader = prepare_fake_loader(model_name=cfg.model.name, n_images=cfg.evaluation.n_images, batch_size=cfg.evaluation.batch_size, collate_fn=collate_fn,
                                        Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                        sobel=cfg.data.sobel, channels=cfg.data.channels)
    for data in tqdm(fake_loader, desc="Generating and computing features from generated images"):
        imgs = data[0].to(cfg.device)
        if conditional:
            batch_conds = data[1].int().view(-1).tolist()
            cond_idxs = [index_mapping.get(item) for item in batch_conds]
            C_FID_IS.update(imgs, cond_idxs, real=False)
        FID_IS.update(imgs, real=False)
        
    print("Computing real_val_test-fake metrics")
    # is_real = FID_IS.compute_IS(real=True)
    # is_fake = FID_IS.compute_IS(real=False)
    # fid = FID_IS.compute_FID()
    
    if conditional:
        fid_matrix_val_test = C_FID_IS.compute_FID_matrix(conditions.tolist(), index="real", column="fake")
        # bcis_real = C_FID_IS.compute_BCIS(real=True)
        # bcis_fake = C_FID_IS.compute_BCIS(real=False)
        # wcis_real = C_FID_IS.compute_WCIS(real=True)
        # wcis_fake = C_FID_IS.compute_WCIS(real=False)

        # bcfid = C_FID_IS.compute_BCFID()
        # wcfid = C_FID_IS.compute_WCFID()

    # print("Real_train-fake evaluation")
    # if conditional:
    #     C_FID_IS = ConditionalFIDandIS(n_conditions=n_train_conditions).to(cfg.device)
    # FID_IS = FIDandIS().to(cfg.device)
    
    # for data in tqdm(train_loader, desc="Loading and computing features from train set"):
    #     imgs = data[0].to(cfg.device)
    #     if conditional:
    #         batch_conds = data[1].int().view(-1).tolist()
    #         cond_idxs = [train_index_mapping.get(item) for item in batch_conds]
    #         C_FID_IS.update(imgs, cond_idxs, real=True)
    #     FID_IS.update(imgs, real=True)
     
    # if cfg.evaluation.sampling == "real_frequencies":
    #     _, weights = np.unique(train_loader.dataset.df[cfg.model.Y], return_counts=True)
    #     weights = torch.tensor(weights, dtype=torch.float)
    # elif cfg.evaluation.sampling == "balanced":
    #     weights = torch.ones(len(train_conditions), dtype=torch.float)

    # collate_fn = GANCollator(netG, conditions=train_conditions, weights=weights, channels=3, sobel=not cfg.data.sobel, n_to_generate=cfg.evaluation.batch_size, gan_sampling="weighted",
    #                          generator_batch_size=cfg.evaluation.batch_size, input_type=input_type, only_fake=True, transforms=None, device=cfg.device)

    # fake_loader = prepare_fake_loader(model_name=cfg.model.name, n_images=cfg.evaluation.n_images, batch_size=cfg.evaluation.batch_size, collate_fn=collate_fn,
    #                                     Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
    #                                     sobel=cfg.data.sobel, channels=cfg.data.channels)
    # for data in tqdm(fake_loader, desc="Generating and computing features from generated images"):
    #     imgs = data[0].to(cfg.device)
    #     if conditional:
    #         batch_conds = data[1].int().view(-1).tolist()
    #         cond_idxs = [train_index_mapping.get(item) for item in batch_conds]
    #         C_FID_IS.update(imgs, cond_idxs, real=False)
    #     FID_IS.update(imgs, real=False)
        
    # print("Computing real_train-fake metrics")
    # # is_real_train = FID_IS.compute_IS(real=True)
    # # is_fake_train = FID_IS.compute_IS(real=False)
    # # fid_train = FID_IS.compute_FID()
    
    # if conditional:
    #     fid_matrix_train = C_FID_IS.compute_FID_matrix(train_conditions.tolist(), index="real", column="fake")
        # bcis_real_train = C_FID_IS.compute_BCIS(real=True)
        # bcis_fake_train = C_FID_IS.compute_BCIS(real=False)
        # wcis_real_train = C_FID_IS.compute_WCIS(real=True)
        # wcis_fake_train = C_FID_IS.compute_WCIS(real=False)

        # bcfid_train = C_FID_IS.compute_BCFID()
        # wcfid_train = C_FID_IS.compute_WCFID()

    # Compute FID on real data using test and validations sets
    # print("Real_val-real_test evaluation")
    # if conditional:
    #     C_FID_IS = ConditionalFIDandIS(n_conditions=n_conditions).to(cfg.device)
    # FID_IS = FIDandIS().to(cfg.device)
    # if cfg.evaluation.use_val:    
    #     for data in tqdm(val_loader, desc="Loading and computing features from validation set"):
    #         imgs = data[0].to(cfg.device)
    #         if conditional:
    #             batch_conds = data[1].int().view(-1).tolist()
    #             cond_idxs = [index_mapping.get(item) for item in batch_conds]
    #             C_FID_IS.update(imgs, cond_idxs, real=True)
    #         FID_IS.update(imgs, real=True)
    # if cfg.evaluation.use_test:
    #     for data in tqdm(test_loader, desc="Loading and computing features from test set"):
    #         imgs = data[0].to(cfg.device)
    #         if conditional:
    #             batch_conds = data[1].int().view(-1).tolist()
    #             cond_idxs = [index_mapping.get(item) for item in batch_conds]
    #             C_FID_IS.update(imgs, cond_idxs, real=False)
    #         FID_IS.update(imgs, real=False)

    # print("Computing real_val-real_test metrics")
    # # fid_real = FID_IS.compute_FID()
    # if conditional:
    #     fid_matrix_real_val_test = C_FID_IS.compute_FID_matrix(conditions.tolist(), index="real_test", column="real_val")
    #     # bcfid_real = C_FID_IS.compute_BCFID()
    #     # wcfid_real = C_FID_IS.compute_WCFID()

    # Compute FID on real data using train set
    # print("Real-real evaluation")
    # if conditional:
    #     C_FID_IS = ConditionalFIDandIS(n_conditions=n_train_test_conditions).to(cfg.device)
    # FID_IS = FIDandIS().to(cfg.device)
    # i=0
    # for data in tqdm(train_loader, desc="Loading and computing features from training set"):
    #     i+=1
    #     if i%2==0:
    #         real_flag = True
    #     else:
    #         real_flag = False
    #     imgs = data[0].to(cfg.device)
    #     if conditional:
    #         batch_conds = data[1].int().view(-1).tolist()
    #         cond_idxs = [train_test_index_mapping.get(item) for item in batch_conds]
    #         C_FID_IS.update(imgs, cond_idxs, real=real_flag)
    #     FID_IS.update(imgs, real=real_flag)
    # if cfg.evaluation.use_val:
    #     i=0  
    #     for data in tqdm(val_loader, desc="Loading and computing features from validation set"):
    #         i+=1
    #         if i%2==0:
    #             real_flag = True
    #         else:
    #             real_flag = False
    #         imgs = data[0].to(cfg.device)
    #         if conditional:
    #             batch_conds = data[1].int().view(-1).tolist()
    #             cond_idxs = [train_test_index_mapping.get(item) for item in batch_conds]
    #             C_FID_IS.update(imgs, cond_idxs, real=real_flag)
    #         FID_IS.update(imgs, real=real_flag)
    # if cfg.evaluation.use_test:
    #     i=0
    #     for data in tqdm(test_loader, desc="Loading and computing features from test set"):
    #         i+=1
    #         if i%2==0:
    #             real_flag = True
    #         else:
    #             real_flag = False
    #         imgs = data[0].to(cfg.device)
    #         if conditional:
    #             batch_conds = data[1].int().view(-1).tolist()
    #             cond_idxs = [train_test_index_mapping.get(item) for item in batch_conds]
    #             C_FID_IS.update(imgs, cond_idxs, real=real_flag)
    #         FID_IS.update(imgs, real=real_flag)
            
    # print("Computing real_train_val_test-real_train_val_test metrics")
    # # fid_real = FID_IS.compute_FID()
    # if conditional:
    #     fid_matrix_real_train = C_FID_IS.compute_FID_matrix(train_test_conditions.tolist(), index="real", column="real")
    #     # bcfid_real = C_FID_IS.compute_BCFID()
    #     # wcfid_real = C_FID_IS.compute_WCFID()

    print("Fake-fake evaluation")
    conditions = np.sort(train_loader.dataset.df[cfg.model.Y].unique()).tolist()
    if cfg.evaluation.use_val:
        conditions += np.sort(val_loader.dataset.df[cfg.model.Y].unique()).tolist()
    if cfg.evaluation.use_test:
        conditions += np.sort(test_loader.dataset.df[cfg.model.Y].unique()).tolist()
    conditions = torch.tensor(list(set(conditions)))
    conditions, _ = torch.sort(conditions)
    index_mapping = {cond: idx for idx, cond in enumerate(conditions.int().tolist())}
    n_conditions = len(conditions)

    if cfg.evaluation.sampling == "real_frequencies":
        _, weights = np.unique(pd.concat([train_loader.dataset.df[cfg.model.Y], val_loader.dataset.df[cfg.model.Y], test_loader.dataset.df[cfg.model.Y]]), return_counts=True)
        weights = torch.tensor(weights, dtype=torch.float)
    elif cfg.evaluation.sampling == "balanced":
        weights = torch.ones(n_conditions, dtype=torch.float)

    collate_fn = GANCollator(netG, conditions=conditions, weights=weights, channels=3, sobel=not cfg.data.sobel, n_to_generate=cfg.evaluation.batch_size, gan_sampling="weighted",
                             generator_batch_size=cfg.evaluation.batch_size, input_type=input_type, only_fake=True, transforms=None, device=cfg.device)

    fake_loader = prepare_fake_loader(model_name=cfg.model.name, n_images=cfg.evaluation.n_images, batch_size=cfg.evaluation.batch_size, collate_fn=collate_fn,
                                      Y=cfg.model.Y, min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, 
                                      sobel=cfg.data.sobel, channels=cfg.data.channels)
    # Compute FID on fake data
    if conditional:
        C_FID_IS = ConditionalFIDandIS(n_conditions=n_conditions).to(cfg.device)
    FID_IS = FIDandIS().to(cfg.device)
    for data in tqdm(fake_loader, desc="Generating and computing features from generated images as fake"):
        imgs = data[0].to(cfg.device)
        if conditional:
            batch_conds = data[1].int().view(-1).tolist()
            cond_idxs = [index_mapping.get(item) for item in batch_conds]
            C_FID_IS.update(imgs, cond_idxs, real=False)
        FID_IS.update(imgs, real=False)
        
    for data in tqdm(fake_loader, desc="Generating and computing features from generated images as real"):
        imgs = data[0].to(cfg.device)
        if conditional:
            batch_conds = data[1].int().view(-1).tolist()
            cond_idxs = [index_mapping.get(item) for item in batch_conds]
            C_FID_IS.update(imgs, cond_idxs, real=True)
        FID_IS.update(imgs, real=True)
        
    print("Computing fake-fake metrics")
    # fid_fake = FID_IS.compute_FID()
    if conditional:
        fid_matrix_fake = C_FID_IS.compute_FID_matrix(conditions.tolist(), index="fake", column="fake")
        # bcfid_fake = C_FID_IS.compute_BCFID()
        # wcfid_fake = C_FID_IS.compute_WCFID()

    # model_metrics = {"sampling":cfg.evaluation.sampling,
    #                  "n_images": cfg.evaluation.n_images, "use_val": cfg.evaluation.use_val, "use_test": cfg.evaluation.use_test,
    #                  "is_real": is_real, "is_fake": is_fake, "bcis_real": bcis_real,
    #                  "bcis_fake": bcis_fake, "wcis_real": wcis_real, "wcis_fake": wcis_fake,
    #                  "fid_real": fid_real, "fid": fid, "fid_fake": fid_fake,
    #                  "bcfid_real": bcfid_real, "bcfid": bcfid, "bcfid_fake": bcfid_fake,
    #                  "wcfid_real": wcfid_real, "wcfid": wcfid, "wcfid_fake": wcfid_fake}
    
    # model_metrics_df = pd.DataFrame([model_metrics])
    # intrinsic_metrics_csv = model_path / f"intrinsic_metrics_val_test_{model_save_name}.csv"
    # model_metrics_df.to_csv(intrinsic_metrics_csv, index=False)

    # model_metrics = {"sampling":cfg.evaluation.sampling,
    #                 "n_images": cfg.evaluation.n_images,
    #                 "is_real": is_real_train, "is_fake": is_fake_train, "bcis_real": bcis_real_train,
    #                 "bcis_fake": bcis_fake_train, "wcis_real": wcis_real_train, "wcis_fake": wcis_fake_train,
    #                 "fid_real": fid_real_train, "fid": fid, "fid_fake": fid_fake,
    #                 "bcfid_real": bcfid_real, "bcfid": bcfid, "bcfid_fake": bcfid_fake,
    #                 "wcfid_real": wcfid_real, "wcfid": wcfid, "wcfid_fake": wcfid_fake}
    
    # model_metrics_df = pd.DataFrame([model_metrics])
    # intrinsic_metrics_csv = model_path / f"intrinsic_metrics_val_test_{model_save_name}.csv"
    # model_metrics_df.to_csv(intrinsic_metrics_csv, index=False)

    # index_label = f"{'augmented' if cfg.data.augmented else 'not-augmented'}_{cfg.evaluation.sampling}_{cfg.evaluation.n_images}"
    # fid_matrix_train_csv = model_path / f"fid_matrix_train_{model_save_name}.csv"
    # fid_matrix_train.to_csv(fid_matrix_train_csv)

    fid_matrix_val_test_csv = model_path / f"fid_matrix_val_test_{model_save_name}.csv"
    fid_matrix_val_test.to_csv(fid_matrix_val_test_csv)

    fid_matrix_fake_csv = model_path / f"fid_matrix_fake_{model_save_name}.csv"
    fid_matrix_fake.to_csv(fid_matrix_fake_csv)
    # fid_matrix_real.to_csv(model_path / f"fid_matrix_real_{model_save_name}.csv", index_label=index_label)

    # fid_matrix_real_train_csv = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / "fid_matrix_real_train.csv"
    # fid_matrix_real_train.to_csv(fid_matrix_real_train_csv)
    # fid_matrix_real_val_test_csv = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / "fid_matrix_real_val_test.csv"
    # fid_matrix_real_val_test.to_csv(fid_matrix_real_val_test_csv)
    
    return fid_matrix_val_test_csv, fid_matrix_fake_csv