#TODO: Train classifier on original images -> results from test set; Train classifier using GAN generated images (different proportions) -> results from test set.
# save results in csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.PATHS import PROJECT_PATH, MODELS_PATH
from src.utils import set_seed
from src.model.utils import load_model
from src.classification_model import get_classifier
from src.data import prepare_test_loader, prepare_val_loader
from src.config import cfg
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import hydra
from tqdm import tqdm

@hydra.main(config_path=PROJECT_PATH.as_posix(), config_name="config", version_base=None)
def evaluate_classifier(cfg):

    set_seed()

    if cfg.model.name == "starGAN":
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / cfg.model.name
    else:
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / f"{cfg.model.name}_{cfg.model.gen_conditioning_mode}_{cfg.model.dis_conditioning_mode}" / (cfg.model.gen_name + "-" + cfg.model.dis_name)
    
    train_concentrations = []
    if cfg.classifier.use_real:
        train_concentrations += cfg.data.train_concentrations
        if cfg.classifier.use_fake:
            train_concentrations += cfg.data.fake_concentrations
            if cfg.classifier.gan_augmentation <= 1:
                classifier_name = f"min_val_f1_real_fake_classifier{'_truncated' if cfg.classifier.truncated else ''}_{cfg.model.save_name}"
            else:
                classifier_name = f"{cfg.model.save_name.split('.')[0]}_classifier_{cfg.classifier.nf}{f'_real_aug_{cfg.classifier.real_augmentation}' if cfg.classifier.real_augmentation > 1 else ''}{f'_gan_aug_{cfg.classifier.gan_augmentation}' if cfg.classifier.gan_augmentation > 1 else ''}.pt"
            
            classifier_path = model_path / "classifier" / cfg.classifier.name
            train_concentrations = list(set(train_concentrations))
            train_concentrations.sort()
            train_concentrations = "_".join([str(float(conc)) for conc in train_concentrations])
        else:
            classifier_name = f"min_val_f1_model_{cfg.classifier.nf}{f'_real_aug_{cfg.classifier.real_augmentation}' if cfg.classifier.real_augmentation > 1 else ''}{f'_gan_aug_{cfg.classifier.gan_augmentation}' if cfg.classifier.gan_augmentation > 1 else ''}.pt"
            classifier_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.classifier.augmented else ''}{'_sobel' if cfg.classifier.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / "classifier" / cfg.classifier.name
            train_concentrations = list(set(train_concentrations))
            train_concentrations.sort()
            train_concentrations = "_".join([str(float(conc)) for conc in train_concentrations])

    elif cfg.classifier.use_fake:
        classifier_name = f"min_val_f1_fake_classifier{'_truncated' if cfg.classifier.truncated else ''}_{cfg.model.save_name}"
        classifier_path = model_path / "classifier" / cfg.classifier.name
        train_concentrations = list(set(cfg.data.fake_concentrations))
        train_concentrations.sort()
        train_concentrations = "_".join([str(float(conc)) for conc in train_concentrations])
    
    classifier_path = classifier_path / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / train_concentrations

    classifier = get_classifier(cfg.classifier.name, cfg.classifier.nf)
    classifier = load_model(classifier, classifier_path, name=classifier_name)
    classifier.to(cfg.device)
    classifier.eval()

    test_loader = prepare_val_loader(min_conc=cfg.data.min_concentration,
                                      max_conc=cfg.data.max_concentration, Y="label",
                                      channels=cfg.classifier.channels, sobel=cfg.classifier.sobel, 
                                      conc_intervals=cfg.data.concentration_intervals,
                                      batch_size=cfg.hyperparameters.batch_size, 
                                      return_file=True, return_img_path=True)
    
    cm_name = f"test_cm_" + classifier_name[:-2] + "png"
    cm_file_name = f"test_cm_file_" + classifier_name[:-2] + "png"
    df_frame_name = f"test_incorrect_frames_"+ classifier_name[:-2] + "csv"
    df_file_name = f"test_incorrect_files_"+ classifier_name[:-2] + "csv"
    df_metrics_name = f"test_metrics_"+ classifier_name[:-2] + "csv"
    
    cm, cm_file, acr, prc, rcl, f1m, df_frame, df_file = get_metrics(classifier, test_loader, device=cfg.device)

    plt.rcParams.update({'font.size': 14})
    cm_fig = ConfusionMatrixDisplay(cm, display_labels=cfg.data.concentration_intervals)
    cm_fig.plot()
    plt.savefig((classifier_path / cm_name).as_posix())

    cm_file_fig = ConfusionMatrixDisplay(cm_file, display_labels=cfg.data.concentration_intervals)
    cm_file_fig.plot()
    plt.savefig((classifier_path / cm_file_name).as_posix())

    df_frame.to_csv((classifier_path / df_frame_name).as_posix(), index=False)

    df_file.to_csv((classifier_path / df_file_name).as_posix(), index=False)

    model_metrics_df = pd.DataFrame([{"accuracy": acr, "precision": prc, "recall": rcl, "f1": f1m}])
    model_metrics_df.to_csv((classifier_path / df_metrics_name).as_posix())

def get_metrics(model, dataloader, device=cfg.device):
    model.eval()

    all_files = []
    all_img_path = []
    final_preds = []
    final_outputs = []
    final_y = []
    for images, labels, file, img_path in tqdm(dataloader, desc="Classifying test dataset"):
        all_files += list(file)
        all_img_path += list(img_path)
        images = images.to(device)
        outputs = model(images).to("cpu")
        final_outputs += outputs.tolist()

        predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
        final_preds += predicted_classes.tolist()
        final_y += labels.tolist()

    incorrect_preds_idx = np.array(final_y) != np.array(final_preds)
    incorrect_img_paths = np.array(all_img_path)[incorrect_preds_idx].tolist()
    incorrect_files = np.array(all_files)[incorrect_preds_idx].tolist()
    incorrect_gt = np.array(final_y)[incorrect_preds_idx].tolist()
    incorrect_preds = np.array(final_preds)[incorrect_preds_idx].tolist()

    df_incorrect_frame_preds = pd.DataFrame({"file": incorrect_img_paths, "file number": incorrect_files,
                                             "pred": incorrect_preds, "label": incorrect_gt})

    final_outputs = torch.tensor(final_outputs)
    file_preds = []
    file_y = []
    for file in set(all_files):
        file_idxs = [i for i, j in enumerate(all_files) if j == file]
        file_output = final_outputs[file_idxs].median(dim=0)[0]
        file_preds.append(torch.max(file_output, 0)[1].item())
        file_y.append(final_y[all_files.index(file)])
    
    incorrect_file_preds_idx = np.array(file_y) != np.array(file_preds)
    incorrect_files = np.array(list(set(all_files)))[incorrect_file_preds_idx].tolist()
    incorrect_file_gt = np.array(file_y)[incorrect_file_preds_idx].tolist()
    incorrect_file_preds = np.array(file_preds)[incorrect_file_preds_idx].tolist()

    df_incorrect_file_preds = pd.DataFrame({"file number": incorrect_files,
                                            "pred": incorrect_file_preds, "label": incorrect_file_gt})

    prc = precision_score(final_y, final_preds, average='macro', zero_division=0)
    rcl = recall_score(final_y, final_preds, average='macro', zero_division=0)
    f1m = f1_score(final_y, final_preds, average='macro')
    acr = accuracy_score(final_y, final_preds)
    cm = confusion_matrix(final_y, final_preds)
    cm_file = confusion_matrix(file_y, file_preds)

    return cm, cm_file, acr, prc, rcl, f1m, df_incorrect_frame_preds, df_incorrect_file_preds

