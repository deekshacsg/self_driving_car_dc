import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D, Dropout
from keras.utils import print_summary
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import TensorBoard
from keras.optimizers import Adam


def keras_model(image_x, image_y):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(image_x, image_y, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Dropout(0.6))

    model.add(Dense(64))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.0001), loss="mse")
    filepath = "Autopilot.h5"
    checkpoint1 = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint1]

    return model, callbacks_list


def loadFromPickle():
    """

    :return: features and labels from the pickle
    """
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("angles", "rb") as f:
        angles = np.array(pickle.load(f))

    return features, angles


def augmentData(features, labels):
    """
    For augmentation of the data

    :param features:
    :param labels:
    :return:
    """
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels


def preprocessFeatures(features):
    """
    Normalize Data
    :param features:
    :return:
    """
    # features=features/255.
    features = features / 127.5 - 1.
    return features


def main():
    features, labels = loadFromPickle()
    features = preprocessFeatures(features)
    # features, labels = augmentData(features, labels) Commented for now
    features, labels = shuffle(features, labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 80, 60, 1)
    test_x = test_x.reshape(test_x.shape[0], 80, 60, 1)
    model, callbacks_list = keras_model(80, 60)
    print_summary(model)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, batch_size=64,
              callbacks=[TensorBoard(log_dir="Self_drive_dc")])
    model.save('Self_drive_dc.h5')


if __name__ == '__main__':
    main()
