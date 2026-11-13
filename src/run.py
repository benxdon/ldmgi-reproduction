import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

from ldmgi import ldmgi_sweep
from metrics import acc
from data import DATASETS

LAMBDA = list(np.logspace(-8, 8, 15)) + [np.inf]
SEEDS = range(20)

if __name__ == "__main__":
    test_data = ['coil-20', 'digits', 'mnist-s', 'mnist-t']
    results = dict()

    for data in test_data:
        X, y = DATASETS[data]()
        c = len(np.unique(y))

        all_ldmgi = ldmgi_sweep(X, c, LAMBDA, SEEDS)
        ldmgi_acc, ldmgi_nmi = [], []
        for a, lamb in enumerate(LAMBDA):
            ldmgi_accs = [acc(y, lab) for lab in all_ldmgi[a]]
            ldmgi_nmis = [nmi(y, lab, average_method="geometric") for lab in all_ldmgi[a]]
            ldmgi_acc.append(ldmgi_accs)
            ldmgi_nmi.append(ldmgi_nmis)

        all_km = [KMeans(c, n_init=1, random_state=s).fit_predict(X) for s in SEEDS]
        km_accs = [acc(y, lab) for lab in all_km]
        km_nmis = [nmi(y, lab, average_method="geometric") for lab in all_km]

        results[data] = {
            "lambda": np.array(LAMBDA),
            "ldmgi_acc": np.array(ldmgi_acc),
            "ldmgi_nmi": np.array(ldmgi_nmi),
            "km_acc": np.array(km_accs),
            "km_nmi": np.array(km_nmis)
        }
        print(f"Done {data}")

    np.save("result/results.npy", results, allow_pickle=True)
