from PIL import Image

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO("runs/segment/train/weights/best.pt")

    # Validate the model
    metrics = model.val()