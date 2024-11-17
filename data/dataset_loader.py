import numpy as np
import os


def load_mnist_data():
    dataset_path = os.path.join("data", "raw", "mnist.npz")

    with np.load(dataset_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    return x_train, y_train
