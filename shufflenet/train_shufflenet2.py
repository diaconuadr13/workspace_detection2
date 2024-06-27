from torchvision.models import shufflenet_v2_x0_5

import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from dataset import CustomDataset
from torchmetrics import F1Score, JaccardIndex, Accuracy
from torchmetrics.detection import IntersectionOverUnion

def convert_xywh_to_xyxy(bbox):
    """
    Convert bounding box from format (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)
    """
    center_x, center_y, width, height = bbox
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    return [x_min, y_min, x_max, y_max]
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


val_img_dir = '../table/valid/images'
val_labels_dir = '../table/valid/labels'

train_img_dir = '../table/train/images'
train_labels_dir = '../table/train/labels'

# Create datasets
val_dataset = CustomDataset(val_img_dir, val_labels_dir, transform=False)
train_dataset = CustomDataset(train_img_dir, train_labels_dir, transform=False)

print(len(val_dataset))

# Create data loaders
batch_size = 1
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_output_neurons = 4
n_classes = 1
model = shufflenet_v2_x0_5(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, n_output_neurons)
# Add a sigmoid activation function
model = nn.Sequential(model, nn.Sigmoid())

model = model.to(device)

# Print the model summary
input_size = (3, 640, 480)
summary(model, input_size)

criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 25
best_val_loss = float('inf')
best_epoch = -1
best_iou = -1

for epoch in range(num_epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

    total_train_loss = 0
    total_val_loss = 0
    correct = 0
    iou = 0

    # Training loop
    model.train()
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets
        boxes = targets[0]['boxes'].to(device)
        label = targets[0]['labels'].to(device)
        outputs = model(imgs)

        loss = criterion(outputs, boxes)

        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.update()

    progress_bar.close()
    avg_train_loss = total_train_loss / len(train_loader)

    # Validation loop
    model.eval()
    total_val_loss = 0
    iou = 0
    f1_score = 0
    accuracy = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets
            boxes = targets[0]['boxes'].to(device)
            label = targets[0]['labels'].to(device)

            outputs = model(imgs)

            loss = criterion(outputs, boxes)


            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        tqdm.write(
            f'Epoch {epoch + 1}/{num_epochs}\n'
            f'Training Metrics:\n'
            f'\tLoss: {avg_train_loss}\n'
            f'Validation Metrics:\n'
            f'\tLoss: {avg_val_loss}\n'
        )

        # Save the model if it's the best one seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_unet_model.pth')
            tqdm.write(f'Saved the best model')

    scheduler.step()

