import os
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import PadIfNeeded
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

def count_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return len(lines)
class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1] in ['.png', '.jpg']]

        self.train_augm = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=15, p=0.25),
            PadIfNeeded(min_height=640, min_width=480, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 480),
            ToTensorV2()
        ], is_check_shapes=False, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        self.valid_augm = A.Compose([
            PadIfNeeded(min_height=640, min_width=480, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 480),
            ToTensorV2()
        ], is_check_shapes=False, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load YOLO annotations
        filename_without_ext = os.path.splitext(img_file)[0]
        annotation_file = filename_without_ext + '.txt'
        annotation_path = os.path.join(self.labels_dir, annotation_file)

        boxes = []
        labels = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                if count_lines(annotation_path) == 1:
                    for line in f.readlines():
                        parts = line.strip().split()
                        label = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append([center_x, center_y, width, height])
                        labels.append(label)

        if len(boxes) == 0:
            boxes = [[1e-7, 1e-7, 1e-7, 1e-7]]
            labels = [0]

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        # Apply transformations
        if self.transform:
            augmented = self.train_augm(image=img, bboxes=boxes, class_labels=labels)
            img = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['class_labels']
        else:
            augmented = self.valid_augm(image=img, bboxes=boxes, class_labels=labels)
            img = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['class_labels']

        # Format boxes and labels for model
        targets = [{
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int32)
        }]

        return img, targets

# Example usage:
# train_dataset = CustomDataset(images_dir='train/images', labels_dir='train/labels')
# valid_dataset = CustomDataset(images_dir='valid/images', labels_dir='valid/labels', transform=False)

# Example for getting an item:
# img, targets = train_dataset[0]
# print(img.shape, targets)
