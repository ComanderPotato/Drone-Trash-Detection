import os
import json
import shutil
import random
from pathlib import Path
import argparse
import yaml
parser = argparse.ArgumentParser(description='User args')
parser.add_argument('--dataset_percentage', type=float, default=1, required=False, help='Percentage of dataset to use')
parser.add_argument('--train_split', type=float, default=0.7, required=False, help='Split to use for the testing set')
parser.add_argument('--val_split', type=float, default=0.2, required=False, help='Split to use for the validation set')
parser.add_argument('--test_split', type=float, default=0.1, required=False, help='Split to use for the train set')

args = parser.parse_args()

BASE_DIR = Path("data")
IMAGES_DIR = BASE_DIR / "images"
ANNOTATIONS_PATH = BASE_DIR / "annotations.json"
SPLITS = {'train': args.train_split, 'val': args.val_split, 'test': args.test_split}


for split in SPLITS:
    split_dir = BASE_DIR / f"{split}"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

with open(ANNOTATIONS_PATH) as f:
    coco = json.load(f)

images = coco['images']
random.shuffle(images)

num_images = len(images)
split_sizes = {
    'train': int(SPLITS['train'] * num_images),
    'val': int(SPLITS['val'] * num_images),
}
split_sizes['test'] = num_images - split_sizes['train'] - split_sizes['val']

split_images = {}
start = 0
for split, size in split_sizes.items():
    split_images[split] = images[start:start + size]
    start += size

annotations_by_image = {}
for ann in coco['annotations']:
    annotations_by_image.setdefault(ann['image_id'], []).append(ann)

for split, imgs in split_images.items():
    new_json = {
        'images': imgs,
        'annotations': [],
        'categories': coco['categories']
    }

    image_ids = set(img['id'] for img in imgs)
    for img in imgs:
        src = IMAGES_DIR / img['file_name']
        dst = BASE_DIR / f"{split}" / img['file_name']
        shutil.copy2(src, dst)

    for img_id in image_ids:
        anns = annotations_by_image.get(img_id, [])
        new_json['annotations'].extend(anns)

    out_path = BASE_DIR / f"instances_{split}.json"
    with open(out_path, "w") as f:
        json.dump(new_json, f)

categories = coco['categories']
class_names = [cat['name'] for cat in categories]

yaml_dict = {
    'train': str((BASE_DIR / 'train')),
    'val': str((BASE_DIR / 'val')),
    'test': str((BASE_DIR / 'test')),
    'nc': len(class_names),
    'names': class_names
}

yaml_path = BASE_DIR / 'taco.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_dict, f)
