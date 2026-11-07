import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.optimize import linear_sum_assignment

from ldmgi import ldmgi

# metrics
def acc(y_true, y_pred):
    '''A metric to analyze the accuracy of the cluster assignments by optimal mapping'''
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred) # flatten to 1d
    c = max(y_true.max(), y_pred.max()) + 1 # get the max number of classes/clusters
    M = np.zeros((c,c)) # init an empty matrix
    for p, t in zip(y_pred, y_true):
        M[p,t] += 1 # build a table, for each image add 1 at (pred, true)
    rows, cols = linear_sum_assignment(-M) # find the optimal matching
    return M[rows, cols].sum() / len(y_true) # find the accuracy

# data init
X, y = load_digits(return_X_y=True)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# prediction
labels_ldmgi = ldmgi(X, c=10)
print(np.bincount(labels_ldmgi))
print(nmi(y, labels_ldmgi, average_method="geometric"))
print(acc(y, labels_ldmgi))

labels_kMeans = KMeans(10, n_init=10, random_state=0).fit_predict(X)
print(np.bincount(labels_kMeans))
print(nmi(y, labels_kMeans, average_method="geometric"))
print(acc(y, labels_kMeans))

print(np.bincount(y))
