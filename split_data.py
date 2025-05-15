import os
import json
import shutil
import random
from pathlib import Path
import argparse
import yaml
from collections import defaultdict
from ultralytics.utils import LOGGER, TQDM
from pycocotools.coco import COCO
import numpy as np
from ultralytics.data.converter import coco80_to_coco91_class, coco91_to_coco80_class, merge_multi_segment

def write_segments(bboxes, segments, use_segments, keypoints, use_keypoints, fn, f):
    with open((fn / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                        )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
                    
def process_annotations(img, anns, use_segments=True, use_keypoints=False):
    h, w = img["height"], img["width"]

    bboxes = []
    segments = []
    keypoints = []
    for ann in anns:
        if ann.get("iscrowd", False):
            continue
        box = np.array(ann["bbox"], dtype=np.float64)
        box[:2] += box[2:] / 2
        box[[0, 2]] /= w
        box[[1, 3]] /= h
        if box[2] <= 0 or box[3] <= 0:
            continue

        cls = ann["category_id"]
        box = [cls] + box.tolist()
        if box not in bboxes:
            bboxes.append(box)
            if use_segments and ann.get("segmentation") is not None:
                if len(ann["segmentation"]) == 0:
                    segments.append([])
                    continue
                elif len(ann["segmentation"]) > 1:
                    s = merge_multi_segment(ann["segmentation"])
                    s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                else:
                    s = [j for i in ann["segmentation"] for j in i]
                    s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                s = [cls] + s
                segments.append(s)
            if use_keypoints and ann.get("keypoints") is not None:
                keypoints.append(
                    box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                )
    return bboxes, segments, keypoints

def convert_to_yolo(data_dir, output_dir, dataset_percentage=1, train_split=0.7, val_split=0.2, test_split=0.1, shuffle=True):
    output_dir = Path(output_dir)
    for p in output_dir / "labels", output_dir / "images":
        p.mkdir(parents=True, exist_ok=False)
    annotation_file = sorted(Path(data_dir).resolve().glob("*.json"))[0]
    BASE_DIR = Path(data_dir)
    IMAGES_DIR = BASE_DIR / "images"
    SPLITS = {'train': train_split, 'val': val_split, 'test': test_split}
    coco = COCO(annotation_file)
    dataset = coco.getImgIds()[: int(len(coco.getImgIds()) * dataset_percentage)]
    if shuffle:
        random.shuffle(dataset)
    N = int(len(dataset))
    split_sizes = {
        'train': int(SPLITS['train'] * N),
        'val': int(SPLITS['val'] * N),
    }
    split_sizes['test'] = N - split_sizes['train'] - split_sizes['val']
    class_names = coco.loadCats(coco.getCatIds())
    class_names = [cat['name'] for cat in class_names]
    yaml_dict = {
        'path': str(output_dir.resolve()),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': len(class_names),
        'names': class_names
    }
    
    start = 0
    for split, size in split_sizes.items():
        output_images_dir = output_dir / "images" / split
        output_labels_dir = output_dir / "labels" / split
        yaml_dict[split] = str(output_images_dir.resolve())
        for p in output_images_dir, output_labels_dir:
            p.mkdir(parents=True, exist_ok=True)
        for img_id in TQDM(dataset[start:start + size], desc=f"Processing {split}"):
            img = coco.loadImgs(img_id)[0]
            file_name = img['file_name']
            anns = coco.loadAnns(coco.getAnnIds(img_id))
            bboxes, segments, keypoints = process_annotations(img, anns, True)
            write_segments(bboxes, segments, True, keypoints, False, output_labels_dir, file_name)
            src = IMAGES_DIR / img['file_name']
            dst = output_images_dir / img['file_name']
            shutil.copy2(src, dst)
        start += size

    yaml_path = output_dir / f'{output_dir.name}.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f)


parser = argparse.ArgumentParser(description='User args')
parser.add_argument('--data_dir', type=str, required=True, help='Path to root data')
parser.add_argument('--output_dir', type=str, required=True, help='Output of YOLO format data')
parser.add_argument('--dataset_percentage', type=float, default=1, required=False, help='Percentage of dataset to use')
parser.add_argument('--train_split', type=float, default=0.7, required=False, help='Split to use for the testing set')
parser.add_argument('--val_split', type=float, default=0.2, required=False, help='Split to use for the validation set')
parser.add_argument('--test_split', type=float, default=0.1, required=False, help='Split to use for the train set')
parser.add_argument('--shuffle', type=bool, default=True, required=False, help='Whether to shuffle the data')

args = parser.parse_args()
data_dir, output_dir, dataset_percentage, train_split, val_split, test_split, shuffle = args.data_dir, args.output_dir, args.dataset_percentage, args.train_split, args.val_split, args.test_split, args.shuffle
convert_to_yolo(data_dir=data_dir, output_dir=output_dir, dataset_percentage=dataset_percentage, train_split=train_split, val_split=val_split, test_split=test_split, shuffle=shuffle)

# def a():
#     BASE_DIR = Path("data")
#     IMAGES_DIR = BASE_DIR / "images"
#     ANNOTATIONS_PATH = BASE_DIR / "annotations.json"
#     SPLITS = {'train': args.train_split, 'val': args.val_split, 'test': args.test_split}


#     for split in SPLITS:
#         split_dir = BASE_DIR / f"{split}"
#         if split_dir.exists():
#             shutil.rmtree(split_dir)
#         split_dir.mkdir(parents=True)

#     with open(ANNOTATIONS_PATH) as f:
#         json_data = json.load(f)

#     images = json_data['images']
#     random.shuffle(images)

#     num_images = len(images)
#     split_sizes = {
#         'train': int(SPLITS['train'] * num_images),
#         'val': int(SPLITS['val'] * num_images),
#     }
#     split_sizes['test'] = num_images - split_sizes['train'] - split_sizes['val']

#     split_images = {}
#     start = 0
#     for split, size in split_sizes.items():
#         split_images[split] = images[start:start + size]
#         start += size

#     annotations_by_image = {}
#     for ann in json_data['annotations']:
#         annotations_by_image.setdefault(ann['image_id'], []).append(ann)

#     for split, imgs in split_images.items():
#         new_json = {
#             'images': imgs,
#             'annotations': [],
#             'categories': json_data['categories']
#         }

#         image_ids = set(img['id'] for img in imgs)
#         for img in imgs:
#             src = IMAGES_DIR / img['file_name']
#             dst = BASE_DIR / f"{split}" / img['file_name']
#             shutil.copy2(src, dst)

#         for img_id in image_ids:
#             anns = annotations_by_image.get(img_id, [])
#             new_json['annotations'].extend(anns)

#         out_path = BASE_DIR / f"instances_{split}.json"
#         with open(out_path, "w") as f:
#             json.dump(new_json, f)

#     categories = json_data['categories']
#     class_names = [cat['name'] for cat in categories]

#     yaml_dict = {
#         'train': 'train',
#         'val': 'val',
#         'test': 'test',
#         'nc': len(class_names),
#         'names': class_names
#     }

#     yaml_path = BASE_DIR / 'taco.yaml'
#     with open(yaml_path, 'w') as f:
#         yaml.dump(yaml_dict, f)

