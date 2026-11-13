import numpy as np
from scipy.optimize import linear_sum_assignment

def acc(y_true, y_pred):
    '''A metric to analyze the accuracy of the cluster assignments by optimal mapping'''
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred) # flatten to 1d
    c = max(y_true.max(), y_pred.max()) + 1 # get the max number of classes/clusters
    M = np.zeros((c,c)) # init an empty matrix
    for p, t in zip(y_pred, y_true):
        M[p,t] += 1 # build a table, for each image add 1 at (pred, true)
    rows, cols = linear_sum_assignment(-M) # find the optimal matching
    return M[rows, cols].sum() / len(y_true) # find the accuracy

