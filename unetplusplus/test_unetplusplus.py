import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure

from results.results_unetplusplus_exp_C.unetplusplus import SmallNestedUNet
import cv2

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
                                      return_num='True')

    # Remove the remaining small regions
    h, w = rectangle_mask.shape
    total_area = h * w
    properties = measure.regionprops(img_label)
    for i, elem in enumerate(properties):
        area = elem.area
        if area < 0.1 * total_area:
            rectangle_mask[img_label == i + 1] = 0

    return rectangle_mask


def predict(image_path, model_path):
    device = 'cpu'

    # Load the model
    model = SmallNestedUNet(num_classes=1)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Load the image for visualization
    img_vis = cv2.imread(image_path)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

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

    output = torch.sigmoid(output)

    # Apply a threshold to the output to get the predicted mask
    # pred_mask = (output > 0.5).float()
    pred_mask = output
    # Post-process the predicted mask using KMeans
    post_processed_mask = post_process_mask(pred_mask)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    ax[0].imshow(img_vis)  # original image
    ax[0].title.set_text('Original Image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(pred_mask[0].squeeze().cpu().numpy(), cmap='gray')  # predicted mask
    ax[1].title.set_text('Predicted Mask')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(post_processed_mask, cmap='gray')  # post-processed mask
    ax[2].title.set_text('Post-Processed Mask')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    plt.show()

    return post_processed_mask

model_path = 'results/results_unetplusplus_exp_C/best_unet_model.pth'
# pred_mask = predict('../dataseg2/valid/images/poza11_Color_png.rf.0d07e2e7b5877acdcea3311e5d8a21a0.jpg', 'best_unet_model.pth')
pred_mask = predict('../../test/poza_Color.png', model_path)
pred_mask = predict('../../test/alttest_Color.png', model_path)
pred_mask = predict('../../test/seg_Color.png', model_path)
pred_mask = predict('../semantic/valid/poza27_Color_png.rf.acce2b40ca351448802d29a3c65aa1a6.jpg', model_path)
