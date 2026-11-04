import numpy as np
from sklearn.datasets import load_digits

from ldmgi import find_cliques, local_laplacian

X, y = load_digits(return_X_y=True)
print(X.shape, y.shape)

X = X / np.linalg.norm(X, axis=1, keepdims=True)

F = find_cliques(X, k=5)
print(F.shape)

Xi = X[F[0]]
Li = local_laplacian(Xi, 1)
print(Li.shape)
