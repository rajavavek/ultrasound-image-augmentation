import torch
from src.config import cfg
from src.PATHS import MODELS_PATH

def load_model(model, model_path=None, name=cfg.model.save_name):

    if not model_path:
        model_path = MODELS_PATH / f"{cfg.data.dataset}{'_augmented' if cfg.data.augmented else ''}{'_sobel' if cfg.data.sobel else ''}" / f"min_{cfg.data.min_concentration}_max_{cfg.data.max_concentration}" / ("_".join(cfg.data.concentration_intervals if cfg.data.concentration_intervals else "")) / f"{cfg.model.name}_{cfg.model.gen_conditioning_mode}_{cfg.model.dis_conditioning_mode}" / (cfg.model.gen_name + "-" + cfg.model.dis_name)

    model.load_state_dict(torch.load(model_path / name))
    model.eval()
    
    return model

def save_model(model, model_path, name="model.pt"):

    model_path = (model_path / name).as_posix()

    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved at {model_path}")