from pathlib import Path
from ultralytics import YOLO
from cjm_pytorch_utils.core import get_torch_device
from configs import default_validation_config



def evaluate_model(model_path: str, inference_config):
    model = YOLO(model_path)
    dir = model_path.split("/")[1]
    PROJECT = str(Path(f"./logs/{dir}").resolve())

    inference_config = {
        **default_validation_config,
        "device": get_torch_device(),
        "project": PROJECT,
        **inference_config,
    }
    model.val(**inference_config)