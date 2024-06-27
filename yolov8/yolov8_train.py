from ultralytics.models.yolo import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')

    # Train the model
    model.train(data='C:\\Users\\Adi\\Desktop\\licenta_buna\\yolov8\\conveyor2\\data.yaml', close_mosaic=15, epochs=100, patience=30)