import os
import json
from PIL import Image


def load_data():
    """

    :return:
    """
    images_dir = "./resources/log"
    labels_dir = "./resources/json_data"
    throttle, angle = [],  []

    for path, dir, files in os.walk(labels_dir):
        for file in files:
            with open(os.path.join(path, file)) as f:
                data = json.load(f)
                angle.append(data['user/angle'])
                throttle.append(data['user/throttle'])

    print(len(throttle), len(angle))


if __name__ == '__main__':
    load_data()
    img = Image.open("./resources/log/0_cam-image_array_.jpg")
    pix_val = list(img.getdata())
    pix_flat = [channel for pixel in pix_val for channel in pixel]
    print(len(pix_val))
    print(img.format, img.size, img.mode)
