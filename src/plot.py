import matplotlib.pyplot as plt
import numpy as np

results = np.load("result/results.npy", allow_pickle=True).item()
datasets = list(results.keys())

fig, axes = plt.subplots(2, len(datasets), figsize=(4 * len(datasets), 7), sharex=True)

for col, data in enumerate(datasets):
    r = results[data]
    lam = r["lambda"][:-1]                      # finite λ only; last entry is ∞
    for row, metric in enumerate(["acc", "nmi"]):
        ax = axes[row, col]
        scores = r[f"ldmgi_{metric}"]           # shape (n_lambda, n_seeds)
        m, s = scores.mean(axis=1), scores.std(axis=1)

        ax.plot(lam, m[:-1], marker="o", label="LDMGI")
        ax.fill_between(lam, m[:-1] - s[:-1], m[:-1] + s[:-1], alpha=0.2)
        ax.axhline(m[-1], ls="--", c="gray", label="λ = ∞ (graph only)")
        ax.axhline(r[f"km_{metric}"].mean(), ls=":", c="red", label="k-means")

        ax.set_xscale("log")
        if row == 0:
            ax.set_title(data)
        if row == 1:
            ax.set_xlabel("λ")
        if col == 0:
            ax.set_ylabel(metric.upper())

axes[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("result/lambda_sweep.png", dpi=200)
