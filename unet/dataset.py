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
    def __init__(self, img_dir, masks_dir, transform=True):
        self.img_dir = img_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.jpg']

        self.train_augm = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=15, p=0.25),
            PadIfNeeded(min_height=640, min_width=480, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 480),
            ToTensorV2()
        ], is_check_shapes=False)

        self.valid_augm = A.Compose([
            PadIfNeeded(min_height=640, min_width=480, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 480),
            ToTensorV2()
        ], is_check_shapes=False)

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
