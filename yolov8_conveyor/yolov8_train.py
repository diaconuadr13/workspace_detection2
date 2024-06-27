from ultralytics.models.yolo import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(data='C:\\Users\\Adi\\Desktop\\licenta_buna\\conveyor3\\data.yaml', epochs=50, close_mosaic=10, batch=8)