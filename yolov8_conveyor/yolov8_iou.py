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

def calculate_iou_yolo_format(bbox1, bbox2, img_width, img_height):
    bbox1 = [
        (bbox1[0] - bbox1[2] / 2) * img_width,  # x1
        (bbox1[1] - bbox1[3] / 2) * img_height,  # y1
        (bbox1[0] + bbox1[2] / 2) * img_width,  # x2
        (bbox1[1] + bbox1[3] / 2) * img_height,  # y2
    ]
    bbox2 = [
        (bbox2[0] - bbox2[2] / 2) * img_width,  # x1
        (bbox2[1] - bbox2[3] / 2) * img_height,  # y1
        (bbox2[0] + bbox2[2] / 2) * img_width,  # x2
        (bbox2[1] + bbox2[3] / 2) * img_height,  # y2
    ]

    # Calculate the intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate the area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of each bounding box
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the area of union
    union_area = bbox1_area + bbox2_area - intersection

    # Avoid division by zero
    if union_area == 0:
        return 0

    # Calculate IoU
    iou = intersection / union_area
    return iou

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
                labels = f.readlines()
                if not labels:  # Check if the file is empty
                    continue
                class_id, x_center, y_center, width, height = map(float, labels[0].split())
                bbox1 = [x_center, y_center, width, height]

            res = model.predict(image_path)
            for r in res:
                bbox2 = r.boxes.xywhn[0]
                print(bbox2)

            iou = calculate_iou_yolo_format(bbox1, bbox2, w, h)
            iou_values.append(iou.cpu().numpy())

    mean_iou = np.mean(iou_values)
    return mean_iou

def iou_and_acc(prediction, ground_truth):
    iou_score = JaccardIndex(task='binary')
    accuracy_score = Accuracy(task='binary')

    iou = iou_score(prediction, ground_truth)
    acc = accuracy_score(prediction, ground_truth)
    return iou, acc


#train 4
model = YOLO('runs/detect/train4/weights/best.pt')
mean_iou = process_images_and_labels(model, '../conveyor2/valid/images', '../conveyor2/valid/labels')
print(f"Mean IoU: {mean_iou}")
