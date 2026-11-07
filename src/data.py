import numpy as np
import os
from PIL import Image
from sklearn.datasets import load_digits, fetch_openml

def normalize(X):
    '''Scale each image to unit lenght'''
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def load_digits_data():
    X, y = load_digits(return_X_y=True)
    return normalize(X),y

def load_usps():
    X, y = fetch_openml("usps", as_frame=False, return_X_y=True)
    return normalize(X),y

def load_mnist():
    X, y = fetch_openml("mnist_784", as_frame=False, return_X_y=True)
    return normalize(X), y

def load_coil20(path="data/coil-20-proc"):
    X, y = [], []
    root, dirs, files = next(os.walk(path))
    for file in sorted(files):
        if not file.endswith(".png"):
            continue
        img = Image.open(os.path.join(root, file)).convert("L").resize((32, 32)) # resize so that it matches the paper
        X.append(np.asarray(img, dtype=float).flatten())
        y.append(int(file.split('__')[0][3:]) - 1)
    return normalize(np.array(X)), np.array(y)

DATASETS = {
    'digits': load_digits_data,
    'usps': load_usps,
    'mnist': load_mnist,
    'coil-20': load_coil20,
}


