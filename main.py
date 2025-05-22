import torch
import gc
import argparse
from pathlib import Path
from cjm_pytorch_utils.core import get_torch_device
from train import train_model
from val import evaluate_model
from tune import tune_model

DEVICE = get_torch_device()

SEED = 123
def clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
def initialise_training(model, data, project_suffix=None):
    custom_training_config = {
        "model": model,
        "data": str(Path(data).resolve()),
        "batch": 4,
        "epochs": 100,
        "cos_lr": True,
        "patience": 25,
        "imgsz": 1024,
        "dropout": 0.3,
        "plots": True,
    }
    custom_augmentation_config = {
        "degrees": 15,
        "mosaic": 0.3,
        "shear": 2.5,
        "fliplr": 0.5,  
        "cutmix": 0.2, 
        "copy_paste": 0.2, 
        "copy_paste_mode": "mixup"
    }
    train_model(custom_training_config, custom_augmentation_config, project_suffix)
def initialise_evaluation(model: str, data: str):

    model_path = f"logs/{model.split('.')}/train/weights/best.pt"
    custom_evaluation_config = {
        "data": str(Path(data).resolve()),
        "plots": True,
        "save_conf": True,
        "save_crop": True,
    }
    evaluate_model(model_path, custom_evaluation_config)

def initialise_tune(model: str, data: str, project_suffix=None):
    custom_training_config = {
        "model": model,
        "data": str(Path(data).resolve()),
        "batch": 4,
        "optimizer": "AdamW",
        "epochs": 50,
        "iterations": 100,
        "cos_lr": True,
        "patience": 5,
        "imgsz": 1024,
        "dropout": 0.3,
        "plots": False,
        "seed": 123,
    }
    custom_augmentation_config = {
        "degrees": 10,
        "translate": 0.2,
        "scale": 0.1,
        "shear": 2.5,
        "perspective": 1e-4,
        "fliplr": 0.4,
        "mosaic": 0.2,
        "mixup": 0.2,
        "copy_paste": 0.5
    }
    search_space = {
        "lr0": (1e-5, 1e-1),
        "lrf": (0.01, 0.5),
        "weight_decay": (3e-5, 5e-3),
        "momentum": (0.8, 0.95),
        "warmup_epochs": (0.0, 5.0),
        "warmup_momentum": (0.0, 0.95),
        "box": (0.02, 0.2),
        "cls": (0.2, 4.0),
        "hsv_h": (0.0, 0.3),
        "hsv_s": (0.0, 0.3),
        "hsv_v": (0.0, 0.3),
        "degrees": (0.0, 20.0),
        "translate": (0.0, 0.5),
        "scale": (0.0, 0.25),
        "shear": (0.0, 5.0),
        "perspective": (0.0, 1e-3),
        "fliplr": (0.0, 1.0),
        "mosaic": (0.0, 0.5),
        "mixup": (0.0, 0.5),
        "copy_paste": (0.0, 1.0)
    }
    tune_model(custom_training_config, custom_augmentation_config, search_space, project_suffix)
if __name__ == "__main__":
    model = "yolo11n-seg.pt"
    data = "taco_yolo/taco_yolo.yaml"
    suffix = "preprocessed_data_tune"
    model.split(".")
    # model.split(".")[0]
    # clear_cuda_memory()
    # initialise_training(model, data, suffix)
    initialise_tune(model, data, suffix)
    # initialise_evaluation(model, data)

