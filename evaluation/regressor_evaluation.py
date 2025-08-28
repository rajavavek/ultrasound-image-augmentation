import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.PATHS import PROJECT_PATH, MODELS_PATH
from src.utils import set_seed
from src.model import load_model
from src.regression_model import get_regressor
from src.data import prepare_test_loader
from src.config import cfg
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import hydra
from tqdm import tqdm

@hydra.main(config_path=PROJECT_PATH.as_posix(), config_name="config", version_base=None)
def evaluate_regressor(cfg):

    set_seed()

    if cfg.model.name == "starGAN":
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / cfg.model.name
    else:
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / f"{cfg.model.name}_{cfg.model.gen_conditioning_mode}_{cfg.model.dis_conditioning_mode}" / (cfg.model.gen_name + "-" + cfg.model.dis_name)
    
    train_concentrations = []
    if cfg.regressor.use_real:
        train_concentrations += cfg.data.train_concentrations
        if cfg.regressor.use_fake:
            train_concentrations += cfg.data.fake_concentrations
            if cfg.regressor.gan_augmentation <= 1:
                regressor_save_name = f"min_val_loss_real_fake_regressor{'_truncated' if cfg.regressor.truncated else ''}_{cfg.model.save_name}"
            else:
                regressor_save_name = f"{cfg.model.save_name.split('.')[0]}_regressor_{cfg.regressor.nf}{f'_real_aug_{cfg.regressor.real_augmentation}' if cfg.regressor.real_augmentation > 1 else ''}{f'_gan_aug_{cfg.regressor.gan_augmentation}' if cfg.regressor.gan_augmentation > 1 else ''}.pt"
            
            regressor_path = model_path / "regressor" / cfg.regressor.name
            train_concentrations = list(set(train_concentrations))
            train_concentrations.sort()
            train_concentrations = "_".join([str(float(conc)) for conc in train_concentrations])
            regressor_path = regressor_path / train_concentrations
        else:
            regressor_save_name = f"min_val_loss_model_{cfg.regressor.nf}{f'_real_aug_{cfg.regressor.real_augmentation}' if cfg.regressor.real_augmentation > 1 else ''}{f'_gan_aug_{cfg.regressor.gan_augmentation}' if cfg.regressor.gan_augmentation > 1 else ''}.pt"
            regressor_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.regressor.augmented else ''}{'_sobel' if cfg.regressor.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / "regressor" / cfg.regressor.name
            train_concentrations = list(set(train_concentrations))
            train_concentrations.sort()
            train_concentrations = "_".join([str(float(conc)) for conc in train_concentrations])
            regressor_path = regressor_path / train_concentrations
    elif cfg.regressor.use_fake:
        regressor_save_name = f"min_val_loss_fake_regressor{'_truncated' if cfg.regressor.truncated else ''}_{cfg.model.save_name}"
        regressor_path = model_path / "regressor" / cfg.regressor.name
        train_concentrations = list(set(cfg.data.fake_concentrations))
        train_concentrations.sort()
        train_concentrations = "_".join([str(float(conc)) for conc in train_concentrations])
        regressor_path = regressor_path / train_concentrations

    df_frame_name = f"test_frames_errors_"+ regressor_save_name[:-2] + "csv"
    df_frame_name_per_conc = f"test_frames_errors_per_conc_"+ regressor_save_name[:-2] + "csv"
    df_file_name = f"test_files_errors_"+ regressor_save_name[:-2] + "csv"
    df_metrics_name = f"test_errors_"+ regressor_save_name[:-2] + "csv"
    preds_hist_name = f"test_preds_hist_"+ regressor_save_name[:-2] + "png"

    regressor = get_regressor(cfg.regressor.name, cfg.regressor.nf)
    regressor = load_model(regressor, regressor_path, name=regressor_save_name)
    regressor.to(cfg.device)
    regressor.eval()

    test_loader = prepare_test_loader(dataset=cfg.data.dataset, split=cfg.data.split, augmented=False, Y="concentration",
                                      channels=cfg.regressor.channels, sobel=cfg.regressor.sobel, 
                                      conc_intervals=cfg.data.concentration_intervals,
                                      batch_size=cfg.hyperparameters.batch_size, 
                                      return_file=True, return_img_path=True)
    
    mse, rmse, mae, absolute_errors_std, mape, df_frame_errors, df_file_errors = get_errors(regressor, test_loader, device=cfg.device)

    n_cols = 3
    n_rows = (len(df_frame_errors["concentration"].unique()) / n_cols).__ceil__()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4+n_rows*1.5, 4.8+n_rows*2))
    axes = axes.flatten()
    fig.suptitle('Histogram of predictions per concentration', fontsize=16)

    for i, conc in enumerate(np.sort(df_frame_errors["concentration"].unique()).tolist()):
        axes[i].set_title(f"Concentration {conc}")
        axes[i].set_xlabel("Predicted concentrations")
        axes[i].set_ylabel("Frequency")

        df_conc = df_frame_errors[df_frame_errors["concentration"] == conc]
        bins = np.arange(df_conc["pred_concentration"].min(), df_conc["pred_concentration"].max() + 2) - 0.5
        axes[i].hist(df_conc["pred_concentration"], bins=bins)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig((regressor_path / preds_hist_name).as_posix())

    df_frame_errors.to_csv((regressor_path / df_frame_name).as_posix(), index=False)

    df_frame_errors_per_conc = pd.merge(df_frame_errors.iloc[:,2:].groupby("concentration", as_index=False).mean(), df_frame_errors.iloc[:,2:].groupby("concentration", as_index=False).std(), on="concentration", suffixes=["_mean", "_std"])
    df_frame_errors_per_conc.to_csv((regressor_path / df_frame_name_per_conc).as_posix(), index=False)

    df_file_errors.to_csv((regressor_path / df_file_name).as_posix(), index=False)

    model_metrics_df = pd.DataFrame([{"mse": mse, "rmse": rmse, "mae": mae, "ae_std": absolute_errors_std, "mape": mape}])
    model_metrics_csv = (regressor_path / df_metrics_name).as_posix()
    model_metrics_df.to_csv(model_metrics_csv, index=False)

    return model_metrics_csv

def get_errors(model, dataloader, device=cfg.device):
    model.eval()

    all_files = []
    all_img_path = []
    final_preds = []
    final_y = []
    for images, concs, file, img_path in tqdm(dataloader, desc="Regression on test dataset"):
        all_files += list(file)
        all_img_path += list(img_path)
        images = images.to(device)
        outputs = model(images).to("cpu")

        final_preds += outputs.view(-1).tolist()
        final_y += concs.tolist()

    final_preds = torch.tensor(final_preds).view(-1)
    final_preds[final_preds<0] = 0

    mse = mean_squared_error(final_y, final_preds.tolist())
    rmse = mean_squared_error(final_y, final_preds.tolist(), squared=False)
    mae = mean_absolute_error(final_y, final_preds.tolist())

    final_y = torch.tensor(final_y).view(-1)
    absolute_errors = (final_y - final_preds).abs()
    absolute_errors_std = absolute_errors.std()

    mod_final_y = final_y.clone()
    mod_final_y[mod_final_y == 0] = 1.0
    percentage_errors = absolute_errors / mod_final_y

    mape = mean_absolute_percentage_error(mod_final_y.tolist(), final_preds.tolist())

    df_frame_errors = pd.DataFrame({"file": all_img_path, "file number": all_files,
                                    "concentration": final_y.tolist(),
                                    "pred_concentration": final_preds.tolist(),
                                    "absolute_error": absolute_errors.tolist(),
                                    "percentage_error": percentage_errors.tolist()})
    
    file_preds_median = []
    file_preds_mean = []
    files = []
    file_y = []
    for file in set(all_files):
        files.append(file)
        file_idxs = [i for i, j in enumerate(all_files) if j == file]
        file_preds_median.append(final_preds[file_idxs].median().item())
        file_preds_mean.append(final_preds[file_idxs].mean().item())
        file_y.append(final_y[all_files.index(file)])
    
    file_y = torch.tensor(file_y).view(-1)
    file_preds_median = torch.tensor(file_preds_median).view(-1)
    file_preds_median[file_preds_median<0] = 0
    file_preds_mean = torch.tensor(file_preds_mean).view(-1)
    file_preds_mean[file_preds_mean<0] = 0

    median_absolute_errors = (file_y - file_preds_median).abs()
    mean_absolute_errors = (file_y - file_preds_mean).abs()

    mod_file_y = file_y.clone()
    mod_file_y[mod_file_y == 0] = 1.0
    median_percentage_errors = median_absolute_errors / mod_file_y
    mean_percentage_errors = mean_absolute_errors / mod_file_y

    df_file_errors = pd.DataFrame({"file number": files, "concentration": file_y.tolist(),
                                   "median_pred_cocentration": file_preds_median.tolist(), 
                                   "mean_pred_concentration": file_preds_mean.tolist(),
                                   "median_absolute_error": median_absolute_errors.tolist(),
                                   "mean_absolute_error": mean_absolute_errors.tolist(),
                                   "median_percentage_error": median_percentage_errors.tolist(),
                                   "mean_percentage_error": mean_percentage_errors.tolist()})

    return mse, rmse, mae, absolute_errors_std.item(), mape, df_frame_errors, df_file_errors