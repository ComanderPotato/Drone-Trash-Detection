import torch
import gc
import argparse
from pathlib import Path
from cjm_pytorch_utils.core import get_torch_device
from train import train_model

DEVICE = get_torch_device()

SEED = 123
def clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
def initialise_training():
    custom_training_config = {
        "model": "yolo11n-seg.pt",
        "data": str(Path("preprocessed_taco/preprocessed_taco.yaml").resolve()),
        "batch": 4,
        "imgsz": 1024,
        "plots": True
    }
    custom_augmentation_config = {
        "degrees": 45
    }
    train_model(custom_training_config, custom_augmentation_config)
if __name__ == "__main__":
    clear_cuda_memory()
    initialise_training()

