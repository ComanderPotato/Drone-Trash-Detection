from ultralytics import YOLO
from pathlib import Path
from pathlib import Path
from ultralytics import YOLO
from cjm_pytorch_utils.core import get_torch_device
import cv2
from ultralytics import YOLO
import torch
from configs import default_training_config, default_augmentation_config
# It has small, occluded, cluttered objects → favor models that are strong with small object detection.
# It has unbalanced classes → models with focal loss or augmentation-friendly architectures perform better.
# It's a real-world, messy dataset, so robustness is key.
device = get_torch_device()
NUM_WORKERS = 8


def tune_model(training_config, augmentation_config, search_space, project_suffix = None):
    model = YOLO(training_config['model'])
    model_name = training_config['model'].split(".")[0]
    PROJECT = str(Path(f"./logs/{model_name}{'' if project_suffix == None else f'_{project_suffix}'}").resolve())
    training_config = {
        **default_training_config,
        "device": get_torch_device(),
        "project": PROJECT,
        **training_config
    }
    augmentation_config = {
        **default_augmentation_config,
        **augmentation_config,
    }
    model.tune(**training_config, **augmentation_config, space=search_space)
# MODEL_TYPE = "yolov8n-seg.pt"
# PROJECT = str(Path(f"./tuning/") / MODEL_TYPE.split(".")[0])
# DATA_PATH = str(Path("./taco/taco.yaml").resolve())
# default_settings = {
#     "model": MODEL_TYPE,
#     "data": DATA_PATH,
#     "project": PROJECT,
#     "optimizer": "AdamW",
#     "epochs": 50,
#     "batch": 16,
#     "iterations": 50,
#     "workers": NUM_WORKERS,
#     "device": device, 
#     "plots": False,
#     "save": False,
#     "val": True,
#     "seed": 123,
# }

