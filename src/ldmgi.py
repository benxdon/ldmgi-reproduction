import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_cliques(X, k):
    '''return an (n,k) array: row i = indices of image i's clique'''
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, indices = nbrs.kneighbors(X)
    return indices

def local_laplacian(Xi, lamb):
    '''Xi: (k,m) array, k images of one clique
    return Li: (k,k) array'''
    k = Xi.shape[0]
    Xi_c = Xi - Xi.mean(axis=0)
    Hk = np.eye(k) - 1
    Li = Hk * np.linalg.inv(Xi_c @ Xi_c.T) * Hk
    return Li
