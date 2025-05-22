default_training_config = {
    "model": None,
    "data": None,
    "epochs": 100,
    "time": None,
    "patience": 100,
    "batch": 16,
    "imgsz": 640,
    "save": True,
    "save_period": -1,
    "cache": False,
    "device": None,
    "workers": 8,
    "project": None,
    "name": None,
    "exist_ok": False,
    "pretrained": True,
    "optimizer": "auto",
    "seed": 123,
    "deterministic": True,
    "single_cls": False,
    "classes": None,
    "rect": False,
    "multi_scale": False,
    "cos_lr": False,
    "close_mosaic": 10,
    "resume": False,
    "amp": True,
    "fraction": 1.0,
    "profile": False,
    "freeze": None,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 2.0,
    "nbs": 64,
    "overlap_mask": True,
    "mask_ratio": 4,
    "dropout": 0.0,
    "val": True,
    "plots": False
}
default_validation_config = {
    "data": None,                     # str: Path to dataset config file (e.g., coco8.yaml)
    "imgsz": 640,                     # int: Size of input images
    "batch": 16,                      # int: Number of images per batch
    "save_json": False,              # bool: Save results to JSON
    "conf": 0.001,                   # float: Confidence threshold for detections
    "iou": 0.6,                      # float: IOU threshold for NMS
    "max_det": 300,                 # int: Max detections per image
    "half": True,                    # bool: Enable FP16 precision
    "device": None,                  # str: Device (e.g., cpu, cuda:0)
    "dnn": False,                    # bool: Use OpenCV DNN module
    "plots": False,                  # bool: Generate and save plots
    "classes": None,                 # list[int]: List of class IDs to evaluate
    "rect": True,                    # bool: Use rectangular inference
    "split": "val",                  # str: Dataset split to use
    "project": None,                 # str: Project directory name
    "name": None,                    # str: Name of validation run
    "verbose": False,                # bool: Show detailed logs
    "save_txt": False,              # bool: Save results in text files
    "save_conf": False,             # bool: Include confidences in save_txt output
    "save_crop": False,             # bool: Save cropped images of detections
    "workers": 8,                    # int: Number of data loading workers
    "augment": False,               # bool: Enable test-time augmentation
    "agnostic_nms": False,          # bool: Use class-agnostic NMS
    "single_cls": False             # bool: Treat all classes as one
}

default_inference_config = {
    "source": "ultralytics/assets",
    "conf": 0.25,
    "iou": 0.7,
    "imgsz": 640,
    "half": False,
    "device": None,
    "batch": 1,
    "max_det": 300,
    "vid_stride": 1,
    "stream_buffer": False,
    "visualize": False,
    "augment": False,
    "agnostic_nms": False,
    "classes": None,
    "retina_masks": False,
    "embed": None,
    "project": None,
    "name": None,
    "stream": False,
    "verbose": True
}
default_output_config = {
    "show": False,
    "save": False,  # or True, depending on context (CLI vs Python)
    "save_frames": False,
    "save_txt": False,
    "save_conf": False,
    "save_crop": False,
    "show_labels": True,
    "show_conf": True,
    "show_boxes": True,
    "line_width": None,
    "font_size": None,
    "font": "Arial.ttf",
    "pil": False,
    "kpt_radius": 5,
    "kpt_line": True,
    "masks": True,
    "probs": True,
    "filename": None,
    "color_mode": "class",
    "txt_color": (255, 255, 255)
}

default_augmentation_config = {
    "hsv_h": 0.015,                  # float, range: 0.0 - 1.0
    "hsv_s": 0.7,                    # float, range: 0.0 - 1.0
    "hsv_v": 0.4,                    # float, range: 0.0 - 1.0
    "degrees": 0.0,                 # float, range: 0.0 - 180
    "translate": 0.1,               # float, range: 0.0 - 1.0
    "scale": 0.5,                   # float, range: >= 0.0
    "shear": 0.0,                   # float, range: -180 - +180
    "perspective": 0.0,             # float, range: 0.0 - 0.001
    "flipud": 0.0,                  # float, range: 0.0 - 1.0
    "fliplr": 0.5,                  # float, range: 0.0 - 1.0
    "bgr": 0.0,                     # float, range: 0.0 - 1.0
    "mosaic": 1.0,                  # float, range: 0.0 - 1.0
    "mixup": 0.0,                   # float, range: 0.0 - 1.0
    "cutmix": 0.0,                  # float, range: 0.0 - 1.0
    "copy_paste": 0.0,              # float, range: 0.0 - 1.0
    "copy_paste_mode": "flip",      # str, options: 'flip', 'mixup'
    "auto_augment": "randaugment",  # str, options: 'randaugment', 'autoaugment', 'augmix'
    "erasing": 0.4                  # float, range: 0.0 - 0.9
}
