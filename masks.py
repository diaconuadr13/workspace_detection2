import json
import os
from pprint import pprint
import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_contour(filepath, input_dir='semantic/train', output_dir='semantic/train_masks', target_size=(640, 480)):
    path_to_images = input_dir
    data = load_json_file(filepath)
    image_name = ''
    mask = None

    # Create a dictionary with the counts of annotations for each image
    annotation_counts = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotation_counts:
            annotation_counts[image_id] = 0
        annotation_counts[image_id] += 1

    # Create a dictionary to map image ids to images
    id_to_image = {image['id']: image for image in data['images']}

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        mask_id = annotation['id']

        # Skip images that have more than one object
        if annotation_counts[image_id] > 1:
            # Delete the image file
            image = id_to_image[image_id]
            image_name = image['file_name']
            img_path = os.path.join(path_to_images, image_name)
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted image at {img_path}")
            else:
                print(f"Failed to delete image at {img_path} because the file does not exist")
            continue

        # If an image has no objects, print a message and skip it
        if annotation_counts[image_id] == 0:
            print(f"Skipping image {image_id} because it has no objects")
            continue

        contours = annotation['segmentation']
        image = id_to_image[image_id]
        image_name = image['file_name']
        width = image['width']
        height = image['height']

        # Load the original image
        img_path = os.path.join(path_to_images, image_name)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        # Check if the image is loaded correctly
        if img is None:
            print(f"Failed to load image at {img_path}")
            continue

        # Create a mask with the same dimensions as the original image
        mask_img = np.ones((height, width), dtype=np.uint8) * 255

        # Iterate over each contour
        for contour in contours:
            # Get the segmentation data and reshape it to the required format for cv2.fillPoly
            segmentation = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))

            # Fill the area defined by the segmentation with black color
            mask_img = cv2.fillPoly(mask_img, [segmentation], color=(0.0,))

        # Resize the mask to the target size
        mask = cv2.resize(mask_img, target_size, interpolation=cv2.INTER_NEAREST)

        # Replace .jpg with .npy in the filename
        filename_without_ext = os.path.splitext(image_name)[0]
        npy_filename = filename_without_ext + '.npy'

        # Draw the contour on the image
        img_with_contour = img.copy()
        for contour in contours:
            for point in np.array(contour, dtype=np.int32).reshape((-1, 1, 2)):
                cv2.circle(img_with_contour, tuple(point[0]), 3, (0, 255, 0), -1)
            cv2.imshow('image', mask)
        # # Visualize the original image, the mask, and the image with the contour
        # fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        # ax[0].imshow(img, cmap='gray')  # original image
        # ax[0].title.set_text('Original Image')
        # ax[1].imshow(mask, cmap='gray')  # mask
        # ax[1].title.set_text('Mask')
        # ax[2].imshow(img_with_contour, cmap='gray')  # image with contour
        # ax[2].title.set_text('Image with Contour')
        # plt.show()

        np.save(os.path.join(output_dir, npy_filename), mask)


def main():
    #get_contour('more_pallets/train/_annotations.coco.json', 'more_pallets/train', 'more_pallets/train_masks')
    get_contour('conveyor_seg/train/_annotations.coco.json', 'conveyor_seg/train', 'conveyor_seg/train_masks')


if __name__ == "__main__":
    main()