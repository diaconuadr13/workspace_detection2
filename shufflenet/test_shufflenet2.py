import time

import time

import cv2
import numpy as np
from matplotlib import patches, pyplot as plt
from torch import nn
from torchvision.models import shufflenet_v2_x0_5, squeezenet1_0
from PIL import Image
import torch
import torchvision.transforms as T

n_output_neurons = 4
n_classes = 1
model = shufflenet_v2_x0_5(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, n_output_neurons)
# Add a sigmoid activation function
model = nn.Sequential(model, nn.Sigmoid())

# model.load_state_dict(torch.load('results/results_squeezenet_exp/best_unet_model.pth'))
model.load_state_dict(torch.load('best_unet_model.pth'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Define the device variable
model = model.to(device)
model.eval()

# Load and preprocess the image
#image_path = 'C:\\Users\\Adi\\Desktop\\licenta_buna\\conveyor\\valid\\images\\148_jpg.rf.cac340774c4b6b6d37d64b0439067f24.jpg'
# image_path = '../../conveior_Color.png'
# image_path = '../../conveior2_Color.png'
image_path = '../../robotsnet_research/corners.png'
# Load the image for visualization
image = cv2.imread(image_path)
img_vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = img_vis.shape

# Load the image for model inference
img = np.array(img_vis) / 255.0
img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)

# Perform inference and measure the time
start_time = time.time()
with torch.no_grad():
    output = model(img_tensor)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference time: {inference_time * 1000} ms")

bbox = output[0].cpu().numpy()
bbox[0] *= width  # center_x
bbox[1] *= height  # center_y
bbox[2] *= width  # width
bbox[3] *= height  # height

print(f'Predicted bounding box: {bbox}')

# Convert the bounding box coordinates back to the format expected by cv2.rectangle
center_x, center_y, bbox_width, bbox_height = bbox
top_left = (int(center_x - bbox_width / 2), int(center_y - bbox_height / 2))
bottom_right = (int(center_x + bbox_width / 2), int(center_y + bbox_height / 2))

# Draw the bounding box using OpenCV
color = (255, 0, 0)
thickness = 2
cv2.rectangle(img_vis, top_left, bottom_right, color, thickness)

# Display the image using matplotlib
plt.imshow(img_vis)
plt.xticks([])
plt.yticks([])
plt.show()