import numpy as np
import os


def load_mnist_data():
    # Path to the MNIST dataset
    dataset_path = os.path.join("data", "raw", "mnist.npz")

    # Load dataset
    with np.load(dataset_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0  # Flatten and normalize
    return x_train, y_train
