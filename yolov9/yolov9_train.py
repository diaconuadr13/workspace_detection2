from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLOv10n model from pretrained weights
    model = YOLO("yolov9s.pt")

    # Train the model
    model.train(data="C:\\Users\\Adi\\Desktop\\licenta_buna\\conveyor2\\data.yaml", epochs=50, batch=8)