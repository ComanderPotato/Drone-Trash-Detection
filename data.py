import os
import cv2
import random
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
from typing import List, Tuple
from ultralytics.utils import TQDM
import albumentations as A
def get_class_distribution_per_subset(label_dir='preprocessed_taco/labels'):
    subsets = ['train', 'val', 'test']
    all_distributions = {}

    for subset in subsets:
        class_counts = defaultdict(int)
        subset_path = os.path.join(label_dir, subset)
        if not os.path.exists(subset_path):
            continue
        
        for file_name in os.listdir(subset_path):
            
            if file_name.endswith('.txt'):
                with open(os.path.join(subset_path, file_name), 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            class_counts[class_id] += 1

        all_distributions[subset] = dict(sorted(class_counts.items()))

    return all_distributions

def plot_class_distributions(distributions):
    subsets = list(distributions.keys())
    num_subsets = len(subsets)

    fig, axs = plt.subplots(1, num_subsets, figsize=(6 * num_subsets, 6), sharey=True)

    if num_subsets == 1:
        axs = [axs]  # make it iterable

    for ax, subset in zip(axs, subsets):
        class_counts = distributions[subset]
        class_ids = list(class_counts.keys())
        counts = list(class_counts.values())

        ax.bar(class_ids, counts, color='skyblue')
        ax.set_title(f'{subset.capitalize()} Set')
        ax.set_xlabel('Class ID')
        ax.set_xticks(class_ids)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    axs[0].set_ylabel('Number of Instances')
    plt.suptitle('Class Distribution per Dataset Split', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.05),
        A.RandomBrightnessContrast(p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.6),
        A.GaussianBlur(blur_limit=3, p=0.1),
        # A.Rotate(limit=15, p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            shear={"x": (-2.5, 2.5), "y": (-2.5, 2.5)},
            border_mode=cv2.BORDER_REPLICATE,
            fit_output=False,
            p=0.5,
        )
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_val_test_transform():
    return A.Compose([
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.Rotate(limit=5, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.1),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def parse_yolo_mask_line(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    parts = list(map(float, line.strip().split()))
    cls = int(parts[0])
    points = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
    return cls, points


def format_yolo_mask_line(cls: int, points: List[Tuple[float, float]]) -> str:
    parts = [str(cls)] + [f"{x:.6f} {y:.6f}" for x, y in points]
    return " ".join(parts)


def oversample_dataset(path: str, threshold: int = 10):
    dataset = "train"
    base_path = Path(path)
    labels_folder = base_path / "labels" / dataset
    images_folder = base_path / "images" / dataset

    class_to_files = {}

    print(f"Indexing classes in {dataset}...")
    for label_file in os.listdir(labels_folder):
        label_path = labels_folder / label_file
        image_name = label_file.replace(".txt", ".jpg")
        image_path = images_folder / image_name
        if not image_path.exists():
            continue

        with open(label_path, "r") as f:
            for line in f:
                cls, _ = parse_yolo_mask_line(line)
                class_to_files.setdefault(cls, []).append((image_path, label_path))

    progbar = tqdm(class_to_files.items())
    for cls, files in progbar:
        current_count = len(files)
        if current_count >= threshold:
            continue

        needed = threshold - current_count

        progbar.desc = f"Oversampling {cls}: {current_count} -> {threshold} (+{needed})"
        for i in range(needed):
            source_img_path, source_lbl_path = random.choice(files)

            image = cv2.imread(str(source_img_path))
            h, w = image.shape[:2]

            with open(source_lbl_path, "r") as f:
                lines = f.readlines()

            labels, polygons = [], []
            for line in lines:
                c, pts = parse_yolo_mask_line(line)
                abs_pts = [(x * w, y * h) for x, y in pts]
                labels.append(c)
                polygons.append(abs_pts)

            try:
                augmented = get_train_transform()(
                    image=image,
                    keypoints=[pt for poly in polygons for pt in poly],
                    class_labels=[c for c, pts in zip(labels, polygons) for _ in pts]
                )
            except Exception as e:
                print(f"Augmentation failed for {source_img_path.name}: {e}")
                continue

            aug_keypoints = augmented['keypoints']
            aug_labels = augmented['class_labels']

            new_polygons = []
            k = 0
            for poly in polygons:
                num_pts = len(poly)
                new_polygons.append(aug_keypoints[k:k+num_pts])
                k += num_pts

            base_name = source_img_path.stem.split("_aug")[0]
            new_name = f"{base_name}_aug_cls{cls}_{random.randint(1000, 9999)}"
            new_image_path = images_folder / f"{new_name}.jpg"
            new_label_path = labels_folder / f"{new_name}.txt"

            cv2.imwrite(str(new_image_path), augmented['image'])
            with open(new_label_path, "w") as f_out:
                for c, pts in zip(labels, new_polygons):
                    # norm_pts = [(x / w, y / h) for x, y in pts]
                    norm_pts = [(min(max(x / w, 0), 1), min(max(y / h, 0), 1)) for x, y in pts]
                    f_out.write(format_yolo_mask_line(c, norm_pts) + "\n")

            class_to_files[cls].append((new_image_path, new_label_path))

def ensure_class_representation(dataset_dir, splits=('train', 'val', 'test'), augment=True):
    image_dir = Path(dataset_dir) / "images"
    label_dir = Path(dataset_dir) / "labels"
    class_to_files = {split: {} for split in splits}
    copy_counter = {}

    for split in splits:
        split_label_dir = label_dir / split
        split_image_dir = image_dir / split

        for lbl_file in os.listdir(split_label_dir):
            lbl_path = split_label_dir / lbl_file
            img_file = lbl_file.replace('.txt', '.jpg')
            img_path = split_image_dir / img_file
            if not img_path.exists():
                continue

            with open(lbl_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_to_files[split].setdefault(class_id, []).append((img_path, lbl_path))

    all_classes = set()
    for split_data in class_to_files.values():
        all_classes.update(split_data.keys())

    for target_split in splits:
        target_classes = set(class_to_files[target_split].keys())
        missing_classes = all_classes - target_classes

        for class_id in missing_classes:
            source_split = next((s for s in splits if class_id in class_to_files[s]), None)
            if source_split is None:
                continue

            image_path, label_path = random.choice(class_to_files[source_split][class_id])
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]

            with open(label_path, "r") as f:
                lines = f.readlines()

            labels = []
            polygons = []

            for line in lines:
                cls, pts = parse_yolo_mask_line(line)
                abs_pts = [(x * w, y * h) for x, y in pts]
                labels.append(cls)
                polygons.append(abs_pts)

            if augment:
                try:
                    transform = get_train_transform() if target_split == 'train' else get_val_test_transform()
                    augmented = transform(
                        image=image,
                        keypoints=[pt for poly in polygons for pt in poly],
                        class_labels=[cls for cls, pts in zip(labels, polygons) for _ in pts]
                    )
                    
                    aug_keypoints = augmented['keypoints']
                    aug_labels = augmented['class_labels']

                    new_polygons = []
                    i = 0
                    for poly in polygons:
                        num_pts = len(poly)
                        new_polygons.append(aug_keypoints[i:i+num_pts])
                        i += num_pts

                    image_to_save = augmented['image']
                    polygons_to_save = new_polygons

                except Exception as e:
                    print(f"Augmentation failed for {image_path.name}: {e}")
                    continue
            else:
                image_to_save = image
                polygons_to_save = polygons
            key = (target_split, class_id)
            copy_counter[key] = copy_counter.get(key, 0) + 1
            index = copy_counter[key]

            base_name = image_path.stem
            new_name = f"{base_name}_copy_{index}_{class_id}"
            new_image_path = image_dir / target_split / f"{new_name}.jpg"
            new_label_path = label_dir / target_split / f"{new_name}.txt"

            cv2.imwrite(str(new_image_path), image_to_save)

            with open(new_label_path, "w") as f_out:
                for cls, pts in zip(labels, polygons_to_save):
                    # norm_pts = [(x / w, y / h) for x, y in pts]
                    norm_pts = [(min(max(x / w, 0), 1), min(max(y / h, 0), 1)) for x, y in pts]
                    f_out.write(format_yolo_mask_line(cls, norm_pts) + "\n")

            # print(f"Added class {class_id} to {target_split} as {new_name}")


from split_data import convert_to_yolo

def prepare_dataset(data_dir, 
    output_dir, 
    annotation_file, 
    dataset_percentage=1, 
    train_split=0.7, 
    val_split=0.2, 
    test_split=0.1, 
    shuffle=True,
    augment_copied_data=True,
    oversampled_threshold=15,
):
    # convert_to_yolo(
    #     data_dir=data_dir, 
    #     output_dir=output_dir, 
    #     annotation_file=annotation_file, 
    #     dataset_percentage=dataset_percentage, 
    #     train_split=train_split, 
    #     val_split=val_split, 
    #     test_split=test_split, 
    #     shuffle=shuffle
    #     )
    # ensure_class_representation(dataset_dir=output_dir, augment=augment_copied_data)
    # oversample_dataset(path=output_dir, threshold=oversampled_threshold)
    dist = get_class_distribution_per_subset(label_dir='taco_yolo/labels')
    plot_class_distributions(dist)
prepare_dataset(data_dir="data", output_dir="taco_yolo",annotation_file="data/annotations.json")
# def prepare_for_even_dist(data, annotation_file, output, min_class_count=3):
#     with open(annotation_file, 'r') as f:
#         coco_data = json.load(f)

#     category_count = defaultdict(int)
#     for ann in coco_data['annotations']:
#         category_count[ann['category_id']] += 1

#     image_id_to_image = {img['id']: img for img in coco_data['images']}
#     image_id_to_anns = defaultdict(list)
#     for ann in coco_data['annotations']:
#         image_id_to_anns[ann['image_id']].append(ann)

#     category_to_image_ids = defaultdict(set)
#     for ann in coco_data['annotations']:
#         category_to_image_ids[ann['category_id']].add(ann['image_id'])

#     new_images = list(coco_data['images'])
#     new_annotations = list(coco_data['annotations'])
#     next_image_id = max(img['id'] for img in new_images) + 1
#     next_ann_id = max(ann['id'] for ann in new_annotations) + 1

#     output_img_dir = os.path.join(output, 'images')
#     os.makedirs(output_img_dir, exist_ok=True)

#     filename_counters = defaultdict(int)

#     for img in coco_data['images']:
#         src_path = os.path.join(data, img['file_name'])
#         dst_path = os.path.join(output_img_dir, img['file_name'])
#         if os.path.exists(src_path):
#             shutil.copy2(src_path, dst_path)
#         else:
#             print(f"Warning: missing source image {src_path}")

#     for cat_id, count in category_count.items():
#         if count >= min_class_count:
#             continue

#         deficit = min_class_count - count
#         source_image_ids = list(category_to_image_ids[cat_id])
#         if not source_image_ids:
#             print(f"Warning: No images found for category {cat_id}, skipping.")
#             continue

#         for _ in range(deficit):
#             src_img_id = random.choice(source_image_ids)
#             src_img = image_id_to_image[src_img_id]
#             src_anns = image_id_to_anns[src_img_id]

#             filename_base, ext = os.path.splitext(src_img['file_name'])
#             filename_counters[filename_base] += 1
#             new_filename = f"{filename_base}_copy{filename_counters[filename_base]}{ext}"

#             new_img = dict(src_img)
#             new_img['id'] = next_image_id
#             new_img['file_name'] = new_filename
#             new_images.append(new_img)

#             for ann in src_anns:
#                 new_ann = dict(ann)
#                 new_ann['id'] = next_ann_id
#                 new_ann['image_id'] = next_image_id
#                 new_annotations.append(new_ann)
#                 next_ann_id += 1

#             src_path = os.path.join(data, src_img['file_name'])
#             dst_path = os.path.join(output_img_dir, new_filename)
#             if os.path.exists(src_path):
#                 shutil.copy2(src_path, dst_path)
#             else:
#                 print(f"Warning: missing image {src_path} during duplication")

#             next_image_id += 1

#     coco_data['images'] = new_images
#     coco_data['annotations'] = new_annotations
#     os.makedirs(output, exist_ok=True)
#     with open(os.path.join(output, 'annotations.json'), 'w') as f:
#         json.dump(coco_data, f)

#     print("Finished duplicating rare classes and copying images.")
