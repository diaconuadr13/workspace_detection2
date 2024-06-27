import os
import random
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import numpy as np


def pick_random_image(img_dir):
    img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1] in ['.png', '.jpg']]
    random_img_file = random.choice(img_files)
    return os.path.join(img_dir, random_img_file)

def visualize_each_augmentation(image_path, augmentations, augmentation_names):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    figure, ax = plt.subplots(nrows=1, ncols=len(augmentations)+1, figsize=(20, 4))  # One row, first column for original image, rest for augmentations

    # Display original image
    ax[0].imshow(image)
    ax[0].set_title("Original")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Display each augmentation
    for i, (augmentation, name) in enumerate(zip(augmentations, augmentation_names), start=1):
        # Apply the augmentation and then resize the image to its original dimensions
        augmented = A.Compose([augmentation, A.Resize(original_height, original_width)])(image=image)
        augmented_image = augmented['image']
        ax[i].imshow(augmented_image)
        ax[i].set_title(name)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
    plt.show()

# Define each augmentation separately
augmentations = [
    A.HorizontalFlip(p=1),  # Flipping
    A.VerticalFlip(p=1),  # Flipping
    A.Rotate(limit=270, p=1),  # Rotation
    A.RandomBrightnessContrast(p=1),  # Brightness Adjustment
    A.RandomScale(scale_limit=0.1, p=1),  # Scaling
    A.RandomResizedCrop(640, 480, scale=(0.5, 1.0), p=1),  # Zooming
    A.GaussNoise(var_limit=(1.0, 1.0), p=1),  # Adding Noise
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),  # Color Jittering
]

augmentation_names = [
    "Horizontal Flip",
    "Vertical Flip",
    "270 Degree Rotation",
    "Brightness Adjustment",
    "Scaling",
    "Zooming",
    "Adding Noise",
    "Color Jittering",
]

# Pick a random image from the train images folder
train_img_dir = 'semantic/train'
random_image_path = pick_random_image(train_img_dir)

# Visualize each augmentation
visualize_each_augmentation(random_image_path, augmentations, augmentation_names)