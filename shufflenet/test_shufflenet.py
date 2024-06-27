import os
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch import nn
from torchvision.models import shufflenet_v2_x0_5
from dualhead import CustomModel  # Import the CustomModel class

# Define the number of output neurons and neurons for classifier and bbox predictor
n_output_neurons = 4
n_classifier_neurons = 1
n_bbox_neurons = 4

# Instantiate the model with the same parameters as in train_sufflenet.py
model = CustomModel(n_output_neurons, n_classifier_neurons, n_bbox_neurons)

model.load_state_dict(torch.load('best_unet_model.pth'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

def load_and_preprocess_image(image_path):
    # Load the image for visualization
    image = cv2.imread(image_path)
    img_vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = img_vis.shape
    # Load the image for model inference
    img = np.array(img_vis) / 255.0
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor, img_vis, height, width

def infer_and_draw_bbox(image_path):
    img_tensor, img_vis, height, width = load_and_preprocess_image(image_path)
    with torch.no_grad():
        class_prediction, bbox_prediction = model(img_tensor)

    if class_prediction[0][0] < 0.5:
        class_predicted = 0
    else:
        class_predicted = 1

    if class_predicted == 1:
        print("No object detected")
        return img_vis
    else:
        print("Object detected")
        bbox = bbox_prediction[0].cpu().numpy()
        bbox[0] *= width  # center_x
        bbox[1] *= height  # center_y
        bbox[2] *= width  # width
        bbox[3] *= height  # height
        # Convert the bounding box coordinates back to the format expected by cv2.rectangle
        center_x, center_y, bbox_width, bbox_height = bbox
        top_left = (int(center_x - bbox_width / 2), int(center_y - bbox_height / 2))
        bottom_right = (int(center_x + bbox_width / 2), int(center_y + bbox_height / 2))
        # Draw the bounding box using OpenCV
        color = (255, 0, 0)
        thickness = 5
        cv2.rectangle(img_vis, top_left, bottom_right, color, thickness)
        return img_vis

# Get a list of all images in the validation directory
images_dir = '../conveyor/valid/images'
all_images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".jpg")]

# Randomly select 16 images
selected_images = random.sample(all_images, 16)

# # Define the path of the image you want to process
# image_path = '../../conveior2_Color.png'
#
# # Perform inference and plot the image with the predicted bounding box
# img = infer_and_draw_bbox(image_path)
#
# # Display the image
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# For each selected image, perform inference and plot the image with the predicted bounding box
fig = plt.figure(figsize=(10, 10))
for i, image_path in enumerate(selected_images, 1):
    img = infer_and_draw_bbox(image_path)
    fig.add_subplot(4, 4, i)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.show()