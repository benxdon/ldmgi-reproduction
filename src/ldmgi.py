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

def spectral_rotation(G, n_iter=100, seed=0):
    '''G: (n, c) eigenvectors. Return labels (n,) array of cluster ids 0..c-1'''
    c = G.shape[1]
    Z = G / np.linalg.norm(G, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.standard_normal((c, c)))

    labels = np.argmax(Z @ R, axis=1)
    for _ in range(n_iter):
        Y = np.eye(c)[labels]
        U, _, Vt = np.linalg.svd(Z.T @ Y)
        R = U @ Vt
        new_labels = np.argmax(Z @ R, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels

def ldmgi(X, c, k=5, lamb=1.0, seed=0):
    cliques = find_cliques(X, k)
    L = build_L(X, cliques, lamb)
    G = spectral_embedding(L, c)
    return spectral_rotation(G, seed=seed)
