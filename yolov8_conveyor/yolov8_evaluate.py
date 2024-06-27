from PIL import Image

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO("runs/detect/train5/weights/best.pt")

    # Validate the model
    results = model.val()
