from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLOv8n model
model = YOLO("runs/detect/train/weights/best.pt")

# results = model('dataset/valid/images')
results = model('C:\\Users\\Adi\\Desktop\\conveior_Color.png')  # results list
# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])

    # Show results to screen (in supported environments)
    r.show()

