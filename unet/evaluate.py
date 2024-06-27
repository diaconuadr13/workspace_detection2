import os

import cv2
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from torch import nn
from torch.utils.data import DataLoader

from results.results_unet_exp_B.unet import UNet
from torchmetrics import F1Score, JaccardIndex, Accuracy

from dataset import CustomDataset

def visualize_masks(ground_truth, predicted):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted, cmap='gray')
    plt.title('Predicted')

    plt.show()
def post_process_mask(pred_mask):
    # Convert the predicted mask to a 2D array and scale to 255
    pred_mask_2d = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # Apply Otsu's thresholding
    _, binary_mask = cv2.threshold(pred_mask_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=3)

    # Find contours in the mask
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the bounding rectangle
    rectangle_mask = np.zeros_like(opening)

    if contours:
        # Find the bounding rectangle for each contour and draw it on the mask
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter out small contours
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(rectangle_mask, [box], 0, (255,), thickness=-1)

    [img_label, nums] = measure.label(rectangle_mask,
                                      return_num='True')  # reface etichetarea pentru imagine rebinarizata

    # Remove the remaining small regions
    h, w = rectangle_mask.shape
    total_area = h * w
    properties = measure.regionprops(img_label)
    for i, elem in enumerate(properties):
        area = elem.area
        if area < 0.1 * total_area:
            rectangle_mask[img_label == i + 1] = 0

    return rectangle_mask



def evaluate(image_path, mask_path, model_path):
    device = 'cpu'
    batch_size = 1
    # Load the model
    model = UNet(n_class=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    val_dataset = CustomDataset(image_path, mask_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    iou_score = JaccardIndex(task='binary').to(device)
    accuracy_score = Accuracy(task='binary').to(device)
    criterion = nn.BCEWithLogitsLoss()

    total_val_loss = 0
    iou = 0
    accuracy = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)

            outputs = (outputs > 0.5).float()
            post_processed_mask = post_process_mask(outputs)

            # Convert the numpy array to a tensor and add an extra dimension
            post_processed_mask = torch.from_numpy(post_processed_mask).float().to(device).unsqueeze(0)

            # Convert tensors to numpy arrays for visualization
            ground_truth = masks.squeeze().cpu().numpy()
            predicted = post_processed_mask.squeeze().cpu().numpy() #*255

            # Visualize the ground truth and the predicted mask
            visualize_masks(ground_truth, predicted)

            loss = criterion(post_processed_mask, masks)
            total_val_loss += loss.item()
            iou += iou_score(post_processed_mask, masks)
            accuracy += accuracy_score(post_processed_mask, masks)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_iou_val = iou / len(val_loader)
    avg_accuracy_val = accuracy / len(val_loader)

    print(f"Validation Loss: {avg_val_loss}")
    print(f"Validation IoU: {avg_iou_val}")
    print(f"Validation Accuracy: {avg_accuracy_val}")


if __name__ == "__main__":
    evaluate('../semantic/valid', '../semantic/valid_masks', 'results/results_unet_exp_B/best_unet_model.pth')
