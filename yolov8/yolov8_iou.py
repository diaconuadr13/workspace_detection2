import os
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from shapely.geometry import Polygon
from torchmetrics import JaccardIndex, Accuracy
from skimage.draw import polygon2mask

from torchmetrics import Accuracy

def calculate_iou2_and_accuracy(ground_truth, predicted, h, w):
    ground_truth_polygon = Polygon(ground_truth)
    predicted_polygon = Polygon(predicted)

    # Create binary mask images from the polygons
    ground_truth_mask = polygon2mask((h, w), ground_truth_polygon.exterior.coords)
    predicted_mask = polygon2mask((h, w), predicted_polygon.exterior.coords)

    # Convert the binary mask images to PyTorch tensors
    ground_truth_tensor = torch.from_numpy(ground_truth_mask)
    predicted_tensor = torch.from_numpy(predicted_mask)

    # Calculate the IoU score
    iou_score = JaccardIndex(task='binary')
    iou = iou_score(predicted_tensor, ground_truth_tensor)

    # Calculate the accuracy
    accuracy_score = Accuracy(task='binary')
    accuracy = accuracy_score(predicted_tensor, ground_truth_tensor)

    return iou.item(), accuracy.item()

def calculate_iou(ground_truth, predicted):
    ground_truth_polygon = Polygon(ground_truth)
    predicted_polygon = Polygon(predicted)
    intersection = ground_truth_polygon.intersection(predicted_polygon).area
    union_area = ground_truth_polygon.union(predicted_polygon).area
    if union_area == 0:
        return 0  # Avoid division by zero

    return intersection / union_area

def process_images_and_labels(model, images_dir, labels_dir):
    iou_values = []
    acc_values = []
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))

            img = Image.open(image_path)
            w, h = img.size

            with open(label_path, "r") as f:
                labels = f.read().splitlines()[0]
                class_id, *poly = labels.split(' ')

            poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)  # Read poly, reshape
            poly *= [w, h]  # Unscale
            poly = poly.astype(int)

            res = model.predict(image_path)
            for r in res:
                if r.masks is None:  # If no masks are detected
                    pred_vector = np.zeros_like(poly)  # Assign a zero vector
                else:
                    pred_vector = r.masks.xy[0]
                    pred_vector = pred_vector.astype('int')
                iou2, acc = calculate_iou2_and_accuracy(pred_vector, poly, h, w)
                iou_values.append(iou2)
                acc_values.append(acc)
                print(iou2, acc)

    mean_iou = np.mean(iou_values)
    mean_acc = np.mean(acc_values)

    return mean_iou, mean_acc

def iou_and_acc(prediction, ground_truth):
    iou_score = JaccardIndex(task='binary')
    accuracy_score = Accuracy(task='binary')

    iou = iou_score(prediction, ground_truth)
    acc = accuracy_score(prediction, ground_truth)
    return iou, acc

# model = YOLO('runs/segment/train4/weights/best.pt')
# model = YOLO('train16_transfer_learning/weights/best.pt')
# model = YOLO('train11/weights/best.pt')
model = YOLO('runs/segment/train2/weights/best.pt')
mean_iou, mean_acc = process_images_and_labels(model, 'conveyor/valid/images', 'conveyor/valid/labels')
print(f"Mean IoU: {mean_iou}")
print(f"Mean Accuracy: {mean_acc}")