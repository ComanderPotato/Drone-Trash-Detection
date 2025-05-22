from pathlib import Path
from ultralytics import YOLO
from cjm_pytorch_utils.core import get_torch_device
from configs import default_training_config, default_augmentation_config



def train_model(training_config, augmentation_config = None, project_suffix=None):
    model = YOLO(training_config['model'])
    model_name = training_config['model'].split(".")[0]
    PROJECT = str(Path(f"./logs/{model_name}{'' if project_suffix == None else f'_{project_suffix}'}").resolve())
    training_config = {
        **default_training_config,
        "device": get_torch_device(),
        "project": PROJECT,
        **training_config,
    }
    augmentation_config = {
        **default_augmentation_config,
        **augmentation_config
    }
    model.train(**training_config, **augmentation_config)