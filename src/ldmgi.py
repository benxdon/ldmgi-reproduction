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
    Hk = np.eye(k) - 1/k
    Li = Hk @ np.linalg.inv(Xi_c @ Xi_c.T + lamb * np.eye(k)) @ Hk
    return Li

def build_L(X, cliques, lamb):
    '''return L: (n,n) array, the sum of every cliques Li'''
    n = X.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        idx = cliques[i]
        Li = local_laplacian(X[idx], lamb)
        L[np.ix_(idx, idx)] += Li
    return L

def spectral_embedding(L, c):
    '''return G*: (n,c) array, the eigenvectors of 2nd to c+1 smallest eigenvalues'''
    values, vectors = np.linalg.eigh(L)
    return vectors[:, 1:c+1]
