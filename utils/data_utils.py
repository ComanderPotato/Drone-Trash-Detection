import torch
from torchvision.datasets import CocoDetection
from pycocotools import mask as coco_mask
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split, DataLoader


class TACOInstanceDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img = Image.open(self.ids[idx]).convert("RGB") if isinstance(img, str) else img

        boxes = []
        labels = []
        masks = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

            rle = coco_mask.frPyObjects(ann['segmentation'], img.height, img.width)
            mask = coco_mask.decode(rle)
            if mask.ndim == 3:
                mask = mask.any(axis=2)
            masks.append(mask)

            iscrowd.append(ann.get("iscrowd", 0))

        # Might need to reshape to be [N, 4], [N] shapes
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.uint8)
        iscrowd = torch.tensor(iscrowd, dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
    
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def prepare_dataset(train_split=0.8, validation_split=0.1, test_split=0.1, batch_size=4, seed=123):
    assert train_split + validation_split + test_split == 1, "Train, validation and test splits must sum to 1" 

    def collate_fn(batch):
        return tuple(zip(*batch)) 
    
    dataset = TACOInstanceDataset(
        root="data/images",
        annFile="data/annotations.json",
        transforms=get_transform()
    )
    N = len(dataset)
    N_TRAIN = int(train_split * N)
    N_VALIDATION = int(validation_split * N)
    N_TEST = N - N_TRAIN - N_VALIDATION

    train_ds, validation_ds, test_ds = random_split(dataset, [N_TRAIN, N_VALIDATION, N_TEST], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, validation_loader, test_loader
