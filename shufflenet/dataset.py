import os
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import PadIfNeeded
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

def count_lines(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return len(lines)
    return 0

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]

        self.train_augm = A.Compose([
            A.RandomCrop(width=480, height=480, p=0.5),  # Random Crop
            A.Rotate(limit=30, p=0.5),  # Random Rotation
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.5),  # Random Transl
            A.HorizontalFlip(p=0.5),  # Flipping
            A.VerticalFlip(p=0.5),  # Flipping
            A.RandomBrightnessContrast(p=0.3),  # Brightness Adjustment
            A.RandomScale(scale_limit=0.1, p=0.5),  # Scaling
            A.RandomResizedCrop(640, 480, scale=(0.5, 1.0), p=0.5),  # Zooming
            A.GaussNoise(var_limit=(1.0, 1.0), p=0.3),  # Adding Noise
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),  # Color Jittering
            PadIfNeeded(min_height=640, min_width=480, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 480),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        self.valid_augm = A.Compose([
            PadIfNeeded(min_height=640, min_width=480, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 480),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

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
                for line in f.readlines():
                    parts = line.strip().split()
                    label = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append([center_x, center_y, width, height])
                    labels.append(label)

        # If there is more than one object in the image, skip this image and get the next one
        if len(boxes) > 1:
            return self.__getitem__((idx + 1) % self.__len__())

        if len(boxes) == 0:
            boxes = [[1e-7, 1e-7, 1e-7, 1e-7]]
            labels = [1]

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

        # If bounding boxes are empty after augmentation, assign a special label indicating no objects
        if len(boxes) == 0:
            boxes = np.array([[1e-7, 1e-7, 1e-7, 1e-7]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # Special label for no objects

        # Format boxes and labels for model
        targets = [{
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }]

        return img, targets