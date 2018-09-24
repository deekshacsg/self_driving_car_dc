import os
import json
import pickle
import numpy as np
from PIL import Image
from collections import defaultdict


def load_data(img_size):
    """
    Create pickle objects of features and labels 
    :param img_size: the new size of the re-sized image
    :return: None
    """
    images_dir = "./resources/log"
    labels_dir = "./resources/json_data"
    features, y_angle, y_throttle = [], [], []

    for path, dir, files in os.walk(images_dir):
        for file in files:

            if file.endswith('.jpg'):
                img_id = file.split('_')[0]
                json_record = "record_" + img_id + ".json"

                # resize and convert to grey scale
                img = Image.open(os.path.join(path, file))
                img = img.resize(img_size).convert('L')
                features.append(list(img.getdata()))

                # get throttle and angle
                with open(os.path.join(labels_dir, json_record)) as f:
                    data = json.load(f)
                    y_angle.append(data['user/angle'])
                    y_throttle.append(data['user/throttle'])


    print("%d features, %d angles, %d throttle" % (len(features), len(y_angle), len(y_throttle)))

    X = np.array(features).astype('float32')
    y_angle = np.array(y_angle).astype('float32')

    with open("features", "wb") as f:
        pickle.dump(X, f)

    with open("angles", "wb") as f:
        pickle.dump(y_angle, f)


if __name__ == '__main__':

    images = load_data((80, 60))

    # img = Image.open("./resources/log/0_cam-image_array_.jpg")
    # print("Input image", img.format, img.size, img.mode)
    # img.show()