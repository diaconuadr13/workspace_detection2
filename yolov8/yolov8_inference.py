from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("runs/segment/train2/weights/best.pt")
# model = YOLO("train16_transfer_learning/weights/best.pt")
# results = model('dataset/valid/images')
# results = model('C:\\Users\\Adi\\Desktop\\test_real_pallet.jpg')  # results list
# results = model('C:\\Users\\Adi\\Desktop\\test\\poza_Color.png')  # results list
results = model('../../conveior_Color.png')
# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()
    # Save results image
    # r.save(filename='results0.jpg')  # saves to defaults './results' folder