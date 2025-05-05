import os.path
import json
import argparse
import random
import copy
import shutil

def extract_split_images(annotation_file, source_base_dir="data", target_subdir="test"):
    """
    Copies images listed in the annotation file to a target directory,
    renaming them using their batch prefix, and updates the annotation
    file accordingly.

    Parameters:
        annotation_file (str): Path to the annotations JSON file.
        source_base_dir (str): Root directory containing batch folders.
        target_subdir (str): Subdirectory name to copy images into (e.g., 'train', 'val', 'test').
    """
    # Load annotations
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    # Prepare target directory
    target_dir = os.path.join(source_base_dir, target_subdir)
    os.makedirs(target_dir, exist_ok=True)

    # Track number of copied files
    count = 0

    for img in annotations['images']:
        original_path = img['file_name']
        batch_name, image_name = original_path.split('/')
        new_file_name = f"{batch_name}_{image_name}"

        src = os.path.join(source_base_dir, batch_name, image_name)
        dst = os.path.join(target_dir, new_file_name)

        # Copy image
        shutil.copy(src, dst)

        # Update file_name in-place
        img['file_name'] = target_subdir + '/' + new_file_name
        count += 1

    # Replace the original annotations with the updated ones
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f)





    # out_ann_path = os.path.join(source_base_dir, f"annotations_0_{target_subdir}.json")
    # with open(out_ann_path, 'w') as f:
    #     json.dump(annotations, f)

    print(f"{count} images copied to '{target_dir}' and annotation file updated.")


parser = argparse.ArgumentParser(description='User args')
parser.add_argument('--dataset_dir', required=True, help='Path to dataset annotations')
parser.add_argument('--test_percentage', type=int, default=10, required=False, help='Percentage of images used for the testing set')
parser.add_argument('--val_percentage', type=int, default=10, required=False, help='Percentage of images used for the validation set')
parser.add_argument('--nr_trials', type=int, default=10, required=False, help='Number of splits')
parser.add_argument('--seed', type=int, default=42, required=False, help='Base random seed for reproducibility')

args = parser.parse_args()

ann_input_path = os.path.join(args.dataset_dir, 'annotations.json')

# Load annotations
with open(ann_input_path, 'r') as f:
    dataset = json.loads(f.read())

anns = dataset['annotations']
scene_anns = dataset['scene_annotations']
imgs = dataset['images']
nr_images = len(imgs)

nr_testing_images = int(nr_images * args.test_percentage * 0.01 + 0.5)
nr_nontraining_images = int(nr_images * (args.test_percentage + args.val_percentage) * 0.01 + 0.5)

for i in range(args.nr_trials):
    random.seed(args.seed + i) # Set seed for reproducibility
    shuffled_imgs = imgs.copy()
    random.shuffle(shuffled_imgs)

    # Initialize dataset templates
    base_set = {
        'info': dataset['info'],
        'images': [],
        'annotations': [],
        'scene_annotations': [],
        'licenses': [],
        'categories': dataset['categories'],
        'scene_categories': dataset['scene_categories'],
    }

    train_set = copy.deepcopy(base_set)
    val_set = copy.deepcopy(base_set)
    test_set = copy.deepcopy(base_set)

    test_set['images'] = shuffled_imgs[0:nr_testing_images]
    val_set['images'] = shuffled_imgs[nr_testing_images:nr_nontraining_images]
    train_set['images'] = shuffled_imgs[nr_nontraining_images:nr_images]

    test_img_ids = {img['id'] for img in test_set['images']}
    val_img_ids = {img['id'] for img in val_set['images']}
    train_img_ids = {img['id'] for img in train_set['images']}

    # Split instance annotations
    for ann in anns:
        if ann['image_id'] in test_img_ids:
            test_set['annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['annotations'].append(ann)

    # Split scene annotations
    for ann in scene_anns:
        if ann['image_id'] in test_img_ids:
            test_set['scene_annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['scene_annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['scene_annotations'].append(ann)

    # Write output files
    ann_train_out_path = os.path.join(args.dataset_dir, f'annotations_{i}_train.json')
    ann_val_out_path   = os.path.join(args.dataset_dir, f'annotations_{i}_val.json')
    ann_test_out_path  = os.path.join(args.dataset_dir, f'annotations_{i}_test.json')

    with open(ann_train_out_path, 'w+') as f:
        f.write(json.dumps(train_set))

    with open(ann_val_out_path, 'w+') as f:
        f.write(json.dumps(val_set))

    with open(ann_test_out_path, 'w+') as f:
        f.write(json.dumps(test_set))

extract_split_images("data/annotations_0_train.json", target_subdir="train")
extract_split_images("data/annotations_0_val.json", target_subdir="val")
extract_split_images("data/annotations_0_test.json", target_subdir="test")

