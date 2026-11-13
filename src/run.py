import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

from ldmgi import ldmgi_sweep
from metrics import acc
from data import DATASETS

LAMBDA = [1e-10, 0.01, 0.1, 1, np.inf]
SEEDS = range(20)

if __name__ == "__main__":
    test_data = ['coil-20', 'digits', 'mnist-s', 'mnist-t']
    for data in test_data:
        X, y = DATASETS[data]()
        c = len(np.unique(y))
        best_acc, best_lambda = float('-inf'), LAMBDA[0]
        print(f"data set {data}")

        all_ldmgi = ldmgi_sweep(X, c, LAMBDA, SEEDS)
        for a, lamb in enumerate(LAMBDA):
            ldmgi_accs = [acc(y, lab) for lab in all_ldmgi[a]]
            ldmgi_nmis = [nmi(y, lab, average_method="geometric") for lab in all_ldmgi[a]]
            print(f"{" " * 5}lambda: {lamb}, accuracy: {np.mean(ldmgi_accs):.3f} ± {np.std(ldmgi_accs):.3f}, nmi: {np.mean(ldmgi_nmis):.3f} ± {np.std(ldmgi_nmis):.3f}")

        all_km = [KMeans(c, n_init=1, random_state=s).fit_predict(X) for s in SEEDS]
        km_accs = [acc(y, lab) for lab in all_km]
        km_nmis = [nmi(y, lab, average_method="geometric") for lab in all_km]
        print(f"{" " * 5}KMeans - accuracy: {np.mean(km_accs):.3f} ± {np.std(km_accs):.3f}, nmi: {np.mean(km_nmis):.3f} ± {np.std(km_nmis):.3f}")
