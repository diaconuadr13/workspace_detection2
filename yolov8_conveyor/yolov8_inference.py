from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("runs/detect/train8/weights/best.pt")

# results = model('dataset/valid/images')
results = model('C:\\Users\\Adi\\Desktop\\conveior_Color.png')
# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    r.show()