import yaml
from pathlib import Path
from ultralytics.data.utils import  visualize_image_annotations



# def visualise_image_annotations(img_id, data_path):
#     yaml_path = sorted(Path(data_path).resolve().glob("*.yaml"))[0]
#     print(yaml_path)
#     with open(yaml_path, "r") as f:
#         yaml_file = yaml.safe_load(f)
#     labels_map = {}
#     for index, name in enumerate(yaml_file['names']):
#         labels_map[index] = name
    # visualize_image_annotations(img_path, label_pth, labels_map)
import os
from pathlib import Path
import yaml

def visualise_image_annotations(img_id, data_path):
    data_path = Path(data_path).resolve()

    yaml_path = sorted(data_path.glob("*.yaml"))[0]
    with open(yaml_path, "r") as f:
        yaml_file = yaml.safe_load(f)

    labels_map = {index: name for index, name in enumerate(yaml_file['names'])}

    img_filename = f"{int(img_id):05}.jpg"

    img_path = None
    for subset in ["train", "val", "test"]:
        candidate = data_path / "images" / subset / img_filename
        if candidate.exists():
            img_path = candidate.resolve()
            break

    if img_path is None:
        raise FileNotFoundError(f"Image with id {img_id} not found in images/train, val, or test.")

    label_path = img_path.as_posix().replace("/images/", "/labels/").replace(".jpg", ".txt")
    label_path = Path(label_path).resolve()

    if not label_path.exists():
        raise FileNotFoundError(f"Label for image {img_path.name} not found at {label_path}.")
    # print(img_path, label_path)
    import matplotlib.pyplot as plt
    import numpy as np
    from ultralytics.utils.plotting import colors
    from PIL import Image
    img = np.array(Image.open(img_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(label_path, encoding="utf-8") as file:
        for line in file:
            line = line.split()
            print(line)
            class_id = float(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])
            # class_id, x_center, y_center, width, height = line.split()
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    fig, ax = plt.subplots(1)
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, True)) 
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none") 
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2] 
        ax.text(x, y - 5, labels_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    plt.show()

# visualise_image_annotations(123, "oversampled_taco/")

def check_mask():
    datasets = ["train", "test", "val"]
    base_path = Path("taco_yolo")
    labels_folder = base_path / "labels"

    for split in datasets:
        sub_folder = labels_folder / split
        for file_name in sub_folder.iterdir():
            if file_name.is_file():
                with open(file_name.resolve(), encoding="utf-8") as file:
                    for line in file:
                        line = line.split()
                        points = line[1: ]
                        for i in range(0, len(points), 2):
                            x = float(points[i])
                            y = float(points[i + 1])
                            # if not (0 <= x <= 1 or 0 <= y <= 1):
                            if (x < 0 or x > 1 or y < 0 or y > 1):
                                print(f"Bad {file_name} {x, y}")
                                # break
                        

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_yolo_annotation(image_path, label_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        for coord in coords:
            if len(coords) == 4:
                # bbox format: cx, cy, w, h
                cx, cy, bw, bh = coords
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(class_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            elif len(coords) >= 6:
                # mask format: x1 y1 x2 y2 ...
                points = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]
                cv2.polylines(image, [np.array(points)], isClosed=True, color=(255, 0, 255), thickness=2)
                cv2.putText(image, str(class_id), points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

# taco\images\train\00099_aug_cls43_3556.jpg

image_path = Path("taco_yolo/images/val/00085_copy_1_13.jpg")
label_path = Path("taco_yolo/labels/val/00085_copy_1_13.txt")
# check_mask()
# visualize_yolo_annotation(image_path, label_path)

image_dir = Path("taco_yolo/images/train")
label_dir = Path("taco_yolo/labels/train")

# # Loop through all images with 'aug' in their name
for image_path in image_dir.glob("*aug*.jpg"):
    label_path = label_dir / (image_path.stem + ".txt")

    if label_path.exists():
        print(label_path)
        visualize_yolo_annotation(image_path, label_path)
    else:
        print(f"Label file not found for image: {image_path.name}")

# train: Scanning D:\Drone-Trash-Detection\taco\labels\train.cache... 1304 images, 0 backgrounds, 10 corrupt: 100%|██████████| 1304/1304 [00:00<?, ?it/s]
# train: D:\Drone-Trash-Detection\taco\images\train\00099_aug_cls43_3556.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0706381]
# train: D:\Drone-Trash-Detection\taco\images\train\00099_aug_cls43_4544.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0706381]
# train: D:\Drone-Trash-Detection\taco\images\train\00109_aug_cls47_2985.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.1175184]
# train: D:\Drone-Trash-Detection\taco\images\train\00213_aug_cls19_3847.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0187759]
# train: D:\Drone-Trash-Detection\taco\images\train\00213_aug_cls19_6189.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0187759]
# train: D:\Drone-Trash-Detection\taco\images\train\00299_aug_cls47_1505.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0351285]
# train: D:\Drone-Trash-Detection\taco\images\train\00299_aug_cls47_5607.jpg: ignoring corrupt image/label: negative label values [-0.0354635]
# train: D:\Drone-Trash-Detection\taco\images\train\01354_aug_cls47_3148.jpg: ignoring corrupt image/label: negative label values [-0.06297851]
# train: D:\Drone-Trash-Detection\taco\images\train\01354_aug_cls47_6884.jpg: ignoring corrupt image/label: negative label values [-0.0804625]
# train: D:\Drone-Trash-Detection\taco\images\train\01381_aug_cls24_8190.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0775255]
# albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
# val: Fast image access  (ping: 0.00.0 ms, read: 2578.4486.7 MB/s, size: 1736.5 KB)
# val: Scanning D:\Drone-Trash-Detection\taco\labels\val.cache... 307 images, 0 backgrounds, 2 corrupt: 100%|██████████| 307/307 [00:00<?, ?it/s]
# val: D:\Drone-Trash-Detection\taco\images\val\00837_copy_1_2.jpg: ignoring corrupt image/label: negative label values [-0.4643915  -0.67715347]
# val: D:\Drone-Trash-Detection\taco\images\val\01216_copy_1_19.jpg: ignoring corrupt image/label: negative label values [-0.806214   -0.6038295  -0.7936775  -0.59904146 -0.727623   -0.6968405     
#  -0.51869    -0.6771685  -0.5263715  -0.7817935  -0.602929   -0.6043935
#  -0.6271055  -0.631819   -0.398691   -0.74525905 -0.70861447 -0.4165315 ]
# val: D:\Drone-Trash-Detection\taco\images\val\00837_copy_1_2.jpg: ignoring corrupt image/label: negative label values [-0.4643915  -0.67715347]
# val: D:\Drone-Trash-Detection\taco\images\val\01216_copy_1_19.jpg: ignoring corrupt image/label: negative label values [-0.806214   -0.6038295  -0.7936775  -0.59904146 -0.727623   -0.6968405     
#  -0.51869    -0.6771685  -0.5263715  -0.7817935  -0.602929   -0.6043935
#  -0.6271055  -0.631819   -0.398691   -0.74525905 -0.70861447 -0.4165315 ]



