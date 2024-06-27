import matplotlib.pyplot as plt
from PIL import Image

def display_images(image_paths, titles):
    assert len(image_paths) == len(titles), "Number of images and titles should be equal"

    fig, ax = plt.subplots(1, len(image_paths), figsize=(20, 5))

    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        ax[i].imshow(img)
        ax[i].title.set_text(titles[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()

image_paths = ['../test/poza_Color.png','myplotppa.png', 'myplotppb.png', 'myplotppc.png']
titles = ['Original', 'UNet++ A', 'UNet++ B', 'UNet++ C']

display_images(image_paths, titles)