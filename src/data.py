import numpy as np
import os
from PIL import Image
from sklearn.datasets import load_digits, fetch_openml

def normalize(X):
    '''Scale each image to unit length'''
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def load_digits_data():
    X, y = load_digits(return_X_y=True)
    return normalize(X),y

def load_usps():
    X, y = fetch_openml("usps", version=2, as_frame=False, return_X_y=True, data_home="data", parser="liac-arff")
    return normalize(X),y.astype(int)-1

def load_mnist():
    X, y = fetch_openml("mnist_784", as_frame=False, return_X_y=True, data_home="data", parser="liac-arff")
    X, y = X[60000:65000], y[60000:65000]
    return normalize(X), y.astype(int)

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

def load_jaffe(path="data/jaffe"):
    X, y = [], []
    root, dirs, files = next(os.walk(path))
    imgs = sorted(file for file in files if file.endswith((".tiff", ".tif")))
    names = sorted({file.split(".")[0] for file in imgs})
    label_of = {names[i] : i for i in range(len(names))}
    for file in imgs:
        if not file.endswith(".tiff"):
            continue
        img = Image.open(os.path.join(root, file)).convert("L").resize((26,26))
        X.append(np.asarray(img, dtype=float).flatten())
        y.append(label_of[file.split(".")[0]])
    return normalize(np.array(X)), np.array(y)

DATASETS = {
    'coil-20': load_coil20,
    'usps': load_usps,
    'mnist': load_mnist,
    'jaffe': load_jaffe,
    'digits': load_digits_data,
}

if __name__ == "__main__":
    for name, loader in DATASETS.items():
        X, y = loader()
        counts = np.bincount(y)
        print(f"{name:8s} X={X.shape}  classes={len(counts)}  per-class min/max={counts.min()}/{counts.max()}")
