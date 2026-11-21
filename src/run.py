import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

from ldmgi import ldmgi_sweep
from metrics import acc
from data import DATASETS

LAMBDA = list(np.logspace(-8, 8, 9)) + [np.inf]
SEEDS = range(20)

if __name__ == "__main__":
    test_data = ['coil-20', 'usps', 'usps-shift', 'mnist-s', 'mnist-t', 'jaffe']
    results = dict()

    for data in test_data:
        X, y = DATASETS[data]()
        c = len(np.unique(y))

        all_ldmgi, objs = ldmgi_sweep(X, c, LAMBDA, SEEDS)
        ldmgi_acc, ldmgi_nmi = [], []
        for a, lamb in enumerate(LAMBDA):
            ldmgi_accs = [acc(y, lab) for lab in all_ldmgi[a]]
            ldmgi_nmis = [nmi(y, lab, average_method="geometric") for lab in all_ldmgi[a]]
            ldmgi_acc.append(ldmgi_accs)
            ldmgi_nmi.append(ldmgi_nmis)


        kms = [KMeans(c, n_init=1, random_state=s).fit(X) for s in SEEDS]
        all_km = [km.labels_ for km in kms]
        km_best = int(np.argmin([km.inertia_ for km in kms]))
        km_accs = [acc(y, lab) for lab in all_km]
        km_nmis = [nmi(y, lab, average_method="geometric") for lab in all_km]

        best_idx = objs.argmin(axis=1)
        best_acc = max(ldmgi_acc[a][best_idx[a]] for a in range(len(LAMBDA)-1))
        best_nmi = max(ldmgi_nmi[a][best_idx[a]] for a in range(len(LAMBDA)-1))

        results[data] = {
            "lambda": np.array(LAMBDA),
            "ldmgi_acc": np.array(ldmgi_acc),
            "ldmgi_nmi": np.array(ldmgi_nmi),
            "km_acc": np.array(km_accs),
            "km_nmi": np.array(km_nmis),
            "objs": objs,
            "best_acc": best_acc, "best_nmi": best_nmi,
            "km_inertia": np.array([km.inertia_ for km in kms]),
            "km_best_acc": km_accs[km_best], "km_best_nmi": km_nmis[km_best],
        }
        print(f"Done {data}")
    os.makedirs("result", exist_ok=True)
    np.save("result/results.npy", results, allow_pickle=True)
