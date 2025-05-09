import torch
from torchvision.datasets import CocoDetection
from pycocotools import mask as coco_mask
from PIL import Image
from torch.utils.data import DataLoader
import os
import numpy as np
from pycocotools.coco import COCO
from torchvision.transforms import v2
import random

class TACODataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, annotation_file, ids, transforms=None):
        self.root = img_folder
        self.annotation_file = annotation_file
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = ids
        
    def __len__(self):
        return len(self.ids)

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        
        id = self.ids[idx]
        image, anns = self._load_image(id), self._load_target(id)
        image_width, image_height = image.size

        bounding_boxes = []
        labels = []
        masks = []
        image_ids = []
        iscrowds = []
        areas = []
        for ann in anns:
            x_min, y_min, box_width, box_height = ann['bbox']
            bounding_box = [x_min, y_min, x_min + box_width, y_min + box_height]
            segmentation = ann['segmentation']
            mask = coco_mask.frPyObjects(segmentation, image_height, image_width)
            mask = coco_mask.decode(mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            bounding_boxes.append(bounding_box)
            masks.append(mask)

            image_ids.append(ann['image_id'])
            labels.append(ann['category_id'])
            iscrowds.append(ann['iscrowd'])
            areas.append(ann['area'])


        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        iscrowd = torch.tensor(iscrowds, dtype=torch.uint8)
        areas = torch.tensor(areas, dtype=torch.float32)
        
        target = {
            "boxes": bounding_boxes,
            "labels": labels,
            "masks": masks,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([id]),
            'area': areas
        }
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
    
    def set_transforms(self, transforms):
        self.transforms = transforms

def get_transform(train=False):
    transforms = []
    
    # transforms.append(v2.Resize(224, 224))
    transforms.append(v2.ToTensor())
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.RandomResizedCrop(size=(224, 224), antialias=True))
    transforms.append(v2.RandomHorizontalFlip(p=0.5))
    if train:
        transforms.append(v2.RandomHorizontalFlip(0.5))
    return v2.Compose(transforms)

def prepare_dataset(train_split=0.8, validation_split=0.1, test_split=0.1, batch_size=4, seed=123):
    assert train_split + validation_split + test_split == 1, "Train, validation and test splits must sum to 1" 

    def collate_fn(batch):
        return tuple(zip(*batch)) 
    
    annotation_file = "data/annotations.json"
    img_folder = "data/images"
    coco = COCO(annotation_file)
    ids = list(sorted(coco.imgs.keys()))

    random.seed(42)
    random.shuffle(ids)
    N = len(ids)
    N_TRAIN = int(train_split * N)
    N_VALIDATION = int(validation_split * N)
    N_TEST = N - N_TRAIN - N_VALIDATION

    train_ids = ids[:N_TRAIN]
    validation_ids = ids[N_TRAIN:N_TRAIN + N_VALIDATION]
    test_ids = ids[N_TRAIN + N_VALIDATION:]
    train_ds = TACODataset(img_folder, annotation_file, transforms=get_transform(train=True), ids=train_ids)
    validation_ds = TACODataset(img_folder, annotation_file, transforms=get_transform(train=False), ids=validation_ids)
    test_ds = TACODataset(img_folder, annotation_file, transforms=get_transform(train=False), ids=test_ids)

    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) 

    return train_loader, validation_loader, test_loader

    # dataset = TACOInstanceDataset(
    #     root="data/images",
    #     annFile="data/annotations.json",
    #     transforms=get_transform()
    # )
    # N = len(dataset)
    # N_TRAIN = int(train_split * N)
    # N_VALIDATION = int(validation_split * N)
    # N_TEST = N - N_TRAIN - N_VALIDATION

    # train_ds, validation_ds, test_ds = random_split(dataset, [N_TRAIN, N_VALIDATION, N_TEST], generator=torch.Generator().manual_seed(seed))

    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # return train_loader, validation_loader, test_loader
