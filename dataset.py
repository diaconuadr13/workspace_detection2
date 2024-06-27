import os

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations import PadIfNeeded
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from skimage.transform import resize


class CustomDataset(Dataset):
    def __init__(self, img_dir, masks_dir, train=True):
        self.img_dir = img_dir
        self.masks_dir = masks_dir
        self.img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.jpg']

        self.train_augm = A.Compose([
            A.HorizontalFlip(p=0.5),  # Flipping
            A.VerticalFlip(p=0.5),  # Flipping
            A.Rotate(limit=270, p=0.5),  # Rotation
            A.RandomBrightnessContrast(p=0.2),  # Brightness Adjustment
            A.RandomScale(scale_limit=0.1, p=0.2),  # Scaling
            A.RandomResizedCrop(640, 480, scale=(0.8, 1.0), p=0.25),  # Zooming
            A.GaussNoise(var_limit=(5.0, 10.0), p=0.1),  # Adding Noise with smaller variance
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),  # Color Jittering
            PadIfNeeded(min_height=640, min_width=480, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 480),
            ToTensorV2()
        ], is_check_shapes=False)

        self.valid_augm = A.Compose([
            PadIfNeeded(min_height=640, min_width=480, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 480),
            ToTensorV2()
        ], is_check_shapes=False)

        self.transform = self.train_augm if train else self.valid_augm


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


        mask = None
        if self.masks_dir:
            filename_without_ext = os.path.splitext(img_file)[0]
            mask_file = filename_without_ext + '.npy'
            # mask_file = img_file + '.npy'
            mask_path = os.path.join(self.masks_dir, mask_file)
            mask = np.load(mask_path).astype(np.float32) / 255.0
        # print("Image shape before augmentation: ", img.shape)
        # print("Mask shape before augmentation: ", mask.shape)
        if self.transform:
            augmented = self.train_augm(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            augmented = self.valid_augm(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        # print("Image shape after augmentation: ", img.shape)
        # print("Mask shape after augmentation: ", mask.shape)
        return img, (1 - mask)
