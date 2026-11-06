import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

from ldmgi import ldmgi

X, y = load_digits(return_X_y=True)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

labels_ldmgi = ldmgi(X, c=10)
print(labels_ldmgi.shape)
print(np.bincount(labels_ldmgi))
print(nmi(y, labels_ldmgi))

labels_kMeans = KMeans(10, n_init=10, random_state=0).fit_predict(X)
print(labels_kMeans.shape)
print(np.bincount(labels_kMeans))
print(nmi(y, labels_kMeans))

print(np.bincount(y))
