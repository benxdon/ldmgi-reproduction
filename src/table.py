'''Print comparison tables (ours vs the paper) from result/results.npy.'''
import numpy as np

# Paper numbers (Yang et al., 2010), LDMGI and K-means columns.
# best = Tables II / III, mean ± std = Tables IV / V
PAPER = {
    "coil-20": {"acc_best": 82.4, "nmi_best": 91.1, "acc_mean": (75.3, 4.9), "nmi_mean": (90.0, 0.7),
                "km_acc_best": 67.9, "km_nmi_best": 77.3, "km_acc_mean": (59.2, 3.8), "km_nmi_mean": (74.5, 2.1)},
    "usps":    {"acc_best": 81.5, "nmi_best": 86.5, "acc_mean": (80.5, 5.6), "nmi_mean": (86.1, 2.3),
                "km_acc_best": 67.4, "km_nmi_best": 61.2, "km_acc_mean": (64.9, 3.6), "km_nmi_mean": (60.6, 1.2)},
    "mnist-t": {"acc_best": 72.8, "nmi_best": 69.2, "acc_mean": (71.5, 3.5), "nmi_mean": (68.9, 1.5),
                "km_acc_best": 56.6, "km_nmi_best": 50.3, "km_acc_mean": (55.3, 2.6), "km_nmi_mean": (49.3, 1.0)},
    "mnist-s": {"acc_best": 77.9, "nmi_best": 73.9, "acc_mean": (76.3, 3.4), "nmi_mean": (73.9, 1.5),
                "km_acc_best": 53.2, "km_nmi_best": 49.5, "km_acc_mean": (53.0, 4.1), "km_nmi_mean": (48.5, 2.0)},
    "jaffe":   {"acc_best": 93.9, "nmi_best": 93.6, "acc_mean": (90.4, 6.0), "nmi_mean": (92.6, 1.4),
                "km_acc_best": 84.0, "km_nmi_best": 88.1, "km_acc_mean": (71.3, 10.7), "km_nmi_mean": (79.2, 6.7)},
}
PAPER["usps-shift"] = PAPER["usps"]  # same data set, pixels shifted to [0, 1]; compared against the same paper row
NAMES = {"coil-20": "COIL-20", "usps": "USPS", "usps-shift": "USPS*",
         "mnist-t": "MNIST-T", "mnist-s": "MNIST-S", "jaffe": "JAFFE"}
ORDER = ["coil-20", "usps", "mnist-t", "mnist-s", "jaffe", "usps-shift"]  # paper's order, extra row last


def ours(r, metric):
    '''our numbers for one dataset and metric ("acc" or "nmi"), in percent'''
    scores = r[f"ldmgi_{metric}"]               # (n_lambda, n_seeds), last row is λ = ∞
    finite = scores[:-1]
    i = finite.mean(axis=1).argmax()            # λ with the best mean, as in Tables IV / V
    km = r[f"km_{metric}"]
    return {
        "best": 100 * r[f"best_{metric}"],
        "mean": (100 * finite[i].mean(), 100 * finite[i].std()),
        "lam": r["lambda"][i],
        "inf": (100 * scores[-1].mean(), 100 * scores[-1].std()),
        "km_mean": (100 * km.mean(), 100 * km.std()),
    }


def pm(x):
    return f"{x[0]:.1f} ± {x[1]:.1f}"


if __name__ == "__main__":
    results = np.load("result/results.npy", allow_pickle=True).item()
    datasets = [d for d in ORDER if d in results]

    for metric, best_tab, mean_tab in [("acc", "II", "IV"), ("nmi", "III", "V")]:
        print(f"\n{metric.upper()} (%)  best = lowest-objective run (paper Table {best_tab}), "
              f"mean ± std at best λ (paper Table {mean_tab})")
        head = f"{'':8s} | {'paper best':>10s} {'ours best':>9s} | {'paper mean':>12s} {'ours mean':>12s} {'λ':>6s} | " \
               f"{'ours λ=∞':>12s} | {'paper KM':>12s} {'ours KM':>12s}"
        print(head)
        print("-" * len(head))
        for d in datasets:
            o, p = ours(results[d], metric), PAPER[d]
            print(f"{NAMES[d]:8s} | {p[f'{metric}_best']:10.1f} {o['best']:9.1f} | "
                  f"{pm(p[f'{metric}_mean']):>12s} {pm(o['mean']):>12s} {o['lam']:6.0e} | "
                  f"{pm(o['inf']):>12s} | {pm(p[f'km_{metric}_mean']):>12s} {pm(o['km_mean']):>12s}")

    print("\n* USPS with pixels shifted from [-1, 1] to [0, 1] (our extra experiment, compared to the same paper row)")
