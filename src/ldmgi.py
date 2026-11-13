import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors

def find_cliques(X: np.ndarray, k: int):
    '''return an (n,k) array: row i = indices of image i's clique'''
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, indices = nbrs.kneighbors(X)
    return indices

def local_laplacian(Xi: np.ndarray, lamb: float):
    '''Xi: (k,m) array, k images of one clique
    return Li: (k,k) array'''
    k = Xi.shape[0]
    Xi_c = Xi - Xi.mean(axis=0)
    Hk = np.eye(k) - 1/k
    if lamb == np.inf:
        return Hk
    Li = Hk @ np.linalg.inv(Xi_c @ Xi_c.T + lamb * np.eye(k)) @ Hk
    return Li

def build_L(X: np.ndarray, cliques: np.ndarray, lamb: float):
    '''return L: (n,n) array, the sum of every cliques Li'''
    n = X.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        idx = cliques[i]
        Li = local_laplacian(X[idx], lamb)
        L[np.ix_(idx, idx)] += Li
    return L

def spectral_embedding(L: np.ndarray, c: int):
    '''return G*: (n,c) array, the eigenvectors of 2nd to c+1 smallest eigenvalues'''
    values, vectors = scipy.linalg.eigh(L, subset_by_index=[1, c])
    return vectors

def spectral_rotation(G: np.ndarray, n_iter:int=100, seed:int=0):
    '''G: (n, c) eigenvectors. Return labels (n,) array of cluster ids 0..c-1'''
    c = G.shape[1]
    norm = np.linalg.norm(G, axis=1, keepdims=True)
    Z = G / np.maximum(1e-12, norm)
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

def ldmgi(X: np.ndarray, c: int, k:int=5, lamb:float=1.0, seed:int=0) -> np.ndarray:
    cliques = find_cliques(X, k)
    L = build_L(X, cliques, lamb)
    G = spectral_embedding(L, c)
    return spectral_rotation(G, seed=seed)

def global_laplacian(X: np.ndarray, lamb: float):
    '''LDMGI with k = n: every clique is the whole data set, so L = n * H (Xc Xc^T + lamb I)^-1 H.
    Built directly (one n x n inverse) instead of n identical inverses through build_L.'''
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    H = np.eye(n) - 1/n
    return n * H @ np.linalg.inv(Xc @ Xc.T + lamb * np.eye(n)) @ H

def ldmgi_sweep(X: np.ndarray, c: int, lambdas: list, seeds: list, k:int = 5) -> np.ndarray:
    '''run ldmgi for every combinations of lambda and seed. return labels of shape (len(lambda), len(seeds), n)'''
    cliques = find_cliques(X, k)
    out = np.zeros((len(lambdas), len(seeds), len(X)), dtype=int)
    for a, lamb in enumerate(lambdas):
        L = build_L(X, cliques, lamb)
        G = spectral_embedding(L, c)
        for b, seed in enumerate(seeds):
            out[a,b] = spectral_rotation(G, seed) # this make it cheaper to run
    return out

