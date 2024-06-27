import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from tqdm import tqdm

from dataset import CustomDataset
from results.results_unet_exp_C.unet import UNet
from torch.optim.lr_scheduler import StepLR

from torchmetrics import F1Score, JaccardIndex, Accuracy

from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary


val_img_dir = '../semantic/valid'
val_masks_dir = '../semantic/valid_masks'

train_img_dir = '../semantic/train'
train_masks_dir = '../semantic/train_masks'

# train_img_dir = '../../poze_fara_background/test_dataset'
# train_masks_dir = '../../poze_fara_background/test_dataset_masks'

# Create datasets
val_dataset = CustomDataset(val_img_dir, val_masks_dir)
train_dataset = CustomDataset(train_img_dir, train_masks_dir)
print(len(val_dataset))
# Create data loaders
batch_size = 4
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create the model
model = UNet(n_class=1)
model = model.to(device)

# Print the model summary
input_size = (3, 640, 480)
summary(model, input_size)

criterion = nn.CrossEntropyLoss() if model.n_class > 1 else nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
num_epochs = 25
best_val_loss = float('inf')
best_epoch = -1
best_iou = -1
best_f1_score = -1
best_accuracy = -1

# Scores
f1score = F1Score(task='binary').to(device)
iou_score = JaccardIndex(task='binary').to(device)
accuracy_score = Accuracy(task='binary').to(device)

# Create a SummaryWriter instance
writer = SummaryWriter('runs/unet_experiment_CC')

for epoch in range(num_epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

    total_train_loss = 0
    total_val_loss = 0
    correct = 0
    iou = 0
    f1_score = 0
    accuracy = 0

    # Training loop
    model.train()
    for imgs, masks in progress_bar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        outputs = model(imgs)

        # Metrics
        loss = criterion(outputs.squeeze(), masks.squeeze())
        total_train_loss += loss.item()
        iou += iou_score(outputs.squeeze(), masks.squeeze())
        f1_score += f1score(outputs.squeeze(), masks.squeeze())
        accuracy += accuracy_score(outputs.squeeze(), masks.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)  # Calculate average training loss for this epoch
    avg_iou = iou / len(train_loader)
    avg_f1_score = f1_score / len(train_loader)
    avg_accuracy = accuracy / len(train_loader)

    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('IoU/train', avg_iou, epoch)
    writer.add_scalar('F1 Score/train', avg_f1_score, epoch)
    writer.add_scalar('Accuracy/train', avg_accuracy, epoch)

    # Validation loop
    model.eval()
    total_val_loss = 0
    iou = 0
    f1_score = 0
    accuracy = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)

            loss = criterion(outputs.squeeze(), masks.squeeze())
            total_val_loss += loss.item()
            iou += iou_score(outputs.squeeze(), masks.squeeze())
            f1_score += f1score(outputs.squeeze(), masks.squeeze())
            accuracy += accuracy_score(outputs.squeeze(), masks.squeeze())

    avg_val_loss = total_val_loss/len(val_loader)
    avg_iou_val = iou / len(val_loader)
    avg_f1_score_val = f1_score / len(val_loader)
    avg_accuracy_val = accuracy / len(val_loader)

    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('IoU/val', avg_iou_val, epoch)
    writer.add_scalar('F1 Score/val', avg_f1_score_val, epoch)
    writer.add_scalar('Accuracy/val', avg_accuracy_val, epoch)

    tqdm.write(
        f'Epoch {epoch + 1}/{num_epochs}\n'
        f'Training Metrics:\n'
        f'\tLoss: {avg_train_loss}\n'
        f'\tAccuracy: {avg_accuracy}\n'
        f'\tIoU: {avg_iou}\n'
        f'\tF1 Score: {avg_f1_score}\n'
        f'Validation Metrics:\n' 
        f'\tLoss: {avg_val_loss}\n'
        f'\tAccuracy: {avg_accuracy_val}\n'
        f'\tIoU: {avg_iou_val}\n'
        f'\tF1 Score: {avg_f1_score_val}'
    )

    # Save the model if it's the best one seen so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        best_iou = avg_iou_val
        best_f1_score = avg_f1_score_val
        best_accuracy = avg_accuracy_val
        torch.save(model.state_dict(), 'best_unet_model.pth')
        tqdm.write(f'Saved the best model')

    scheduler.step()

# Save the report
with open('results/results_unet_exp_C/model_report2.txt', 'w') as f:
    f.write(f'Model: UNet\n')
    f.write(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')
    f.write(f'Best validation loss: {best_val_loss}\n')
    f.write(f'Best epoch: {best_epoch}\n')
    f.write(f'Best IoU: {best_iou}\n')
    f.write(f'Best F1 Score: {best_f1_score}\n')
    f.write(f'Best Accuracy: {best_accuracy}\n')
    f.write(f'Number of epochs: {num_epochs}\n')

writer.close()