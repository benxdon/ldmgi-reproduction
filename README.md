# LDMGI reproduction

A from-scratch reproduction of

> Yi Yang, Dong Xu, Feiping Nie, Shuicheng Yan and Yueting Zhuang.
> **Image Clustering Using Local Discriminant Models and Global Integration.**
> *IEEE Transactions on Image Processing*, 19(10):2761–2773, 2010.
> doi:[10.1109/TIP.2010.2049235](https://doi.org/10.1109/TIP.2010.2049235)

Course project for COMP3314 (Machine Learning), The University of Hong Kong.

LDMGI clusters images by fitting a small LDA-style discriminant model on every image's
k-nearest-neighbour clique, summing these local models into one graph Laplacian `L`, and
taking the eigenvectors of `L` followed by spectral rotation to get cluster labels.

## What we did

- Implemented LDMGI from scratch (no official code is available).
- Reproduced the LDMGI and K-means results of the paper's Tables II and IV on five of its ten
  datasets: COIL-20, USPS, MNIST-T, MNIST-S and JAFFE.
- **Ablation:** λ = ∞ removes the discriminant term (`L_i = H_k`), leaving spectral clustering
  on the kNN graph, to measure what the local discriminant models contribute.
- **Preprocessing check:** USPS with pixels shifted from [-1, 1] to [0, 1] (`usps-shift`).
- Welch t-tests (20 runs each) for LDMGI vs K-means and LDMGI vs λ = ∞.

## Setup

Requires **Python 3.12 or newer**.

### Option A: with uv (recommended)

If you have [uv](https://docs.astral.sh/uv/), install everything from the lock file:

```bash
uv sync
```

Then prefix every command below with `uv run`, e.g. `uv run python src/run.py`.

### Option B: with pip

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy scikit-learn matplotlib pandas
```

Then run the commands below with plain `python`, e.g. `python src/run.py`.

### Datasets

USPS and MNIST are downloaded automatically from OpenML on first use (into `data/openml/`).
COIL-20 and JAFFE must be downloaded by hand and placed like this:

```
data/
├── coil-20-proc/     # COIL-20 "processed" images: obj1__0.png … obj20__71.png (1440 files)
└── jaffe/            # JAFFE images: KA.AN1.39.tiff, … (213 .tiff files)
```

- **COIL-20:** Columbia Object Image Library, processed version (`coil-20-proc.zip`), from
  <https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php>.
- **JAFFE:** The Japanese Female Facial Expression database, from its official distribution
  (requires agreeing to its terms of use).

To check that every dataset loads:

```bash
python src/data.py
```

| Name | Images | Size | Classes | Preprocessing |
|---|---|---|---|---|
| `coil-20` | 1440 | 32×32 | 20 | resized to 32×32 |
| `usps` | 9298 | 16×16 | 10 | pixels in [-1, 1], as on OpenML |
| `usps-shift` | 9298 | 16×16 | 10 | pixels shifted to [0, 1] |
| `mnist-t` | 5000 | 28×28 | 10 | first 5000 images of the MNIST test set |
| `mnist-s` | 6996 | 28×28 | 10 | random 1-in-10 sample per digit (seed 0) |
| `jaffe` | 213 | 26×26 | 10 | resized to 26×26 |

Every image is scaled to unit length.

## Running

Run all commands from the repository root (add `uv run` in front if you use uv).

1. **Run the experiments.** This sweeps λ ∈ {10⁻⁸, 10⁻⁶, …, 10⁸, ∞} × 20 seeds on every dataset,
   runs K-means with 20 seeds, and saves everything to `result/results.npy` (the folder is
   created automatically). USPS (n = 9298) is by far the slowest, so the full run takes a while.

   ```bash
   python src/run.py
   ```

   To run a subset, edit `test_data` in `src/run.py` (for a quick test, use `['jaffe']`).

2. **Print the tables and t-tests** (ours vs the paper):

   ```bash
   python src/table.py
   ```

3. **Plot accuracy and NMI against λ** (saved to `result/lambda_sweep.png`):

   ```bash
   python src/plot.py
   ```

All seeds are fixed, so re-running gives identical numbers.

## Code

| File | Contents |
|---|---|
| `src/ldmgi.py` | LDMGI: kNN cliques, local Laplacians `L_i` (Eq. 15), global `L` (Eq. 18), spectral embedding (Eq. 21–22), spectral rotation (Eq. 23–24), and the λ sweep |
| `src/data.py` | Dataset loaders and preprocessing |
| `src/metrics.py` | Clustering accuracy with Hungarian matching (Eq. 36) |
| `src/run.py` | Runs the experiments and saves `result/results.npy` |
| `src/table.py` | Comparison tables against the paper and Welch t-tests |
| `src/plot.py` | λ-sweep figure |

## Results

Clustering accuracy (%). *Best*: the run with the lowest spectral-rotation objective for each λ,
then the best over λ (paper Table II). *Mean*: 20 runs at the λ with the highest mean
(paper Table IV).

| Dataset | Paper best | Ours best | Paper mean | Ours mean | Ours λ = ∞ | Paper K-means | Ours K-means |
|---|---|---|---|---|---|---|---|
| COIL-20 | 82.4 | 85.1 | 75.3 ± 4.9 | 83.1 ± 5.2 | 75.1 | 59.2 | 59.7 |
| USPS | 81.5 | 81.6 | 80.5 ± 5.6 | 80.5 ± 2.3 | 66.9 | 64.9 | 63.4 |
| MNIST-T | 72.8 | 73.6 | 71.5 ± 3.5 | 72.3 ± 4.1 | 58.5 | 55.3 | 54.2 |
| MNIST-S | 77.9 | 71.8 | 76.3 ± 3.4 | 71.2 ± 2.6 | 66.1 | 53.0 | 54.2 |
| JAFFE | 93.9 | 92.0 | 90.4 ± 6.0 | 90.4 ± 7.3 | 83.7 | 71.3 | 86.9 |
| USPS (shifted) | 81.5 | 96.7 | 80.5 ± 5.6 | 95.7 ± 4.3 | 81.4 | 64.9 | 70.7 |

Main findings:

- **The paper's results reproduce.** USPS matches almost exactly, and LDMGI is significantly
  better than K-means on COIL-20, USPS and both MNIST sets (Welch t-test, p < 10⁻¹⁵).
- **JAFFE is the exception.** Our K-means uses k-means++ initialisation and reaches 86.9 (paper:
  71.3), so LDMGI is not significantly better (p = 0.17). With random initialisation K-means
  drops to 75.6 and LDMGI is significantly better again.
- **The discriminant term matters.** λ = ∞ (graph only) costs 5–14 accuracy points, significant
  on every dataset (p < 0.01).
- **Preprocessing matters even more.** Shifting USPS pixels to [0, 1] raises LDMGI from 80.5 to
  95.7, more than the paper's reported margin over its best competitor.
- MNIST-S differs by about 5 points, likely because our random subsample differs from the
  authors'.

## Our work and reused code

All code in `src/` was written by the authors of this project, following the equations in the
paper. No code from the original authors was used. We rely on standard libraries for:

- `sklearn.neighbors.NearestNeighbors`: the kNN search for the cliques
- `sklearn.cluster.KMeans`: the K-means baseline (default k-means++ initialisation)
- `sklearn.metrics.normalized_mutual_info_score`: NMI
- `sklearn.datasets.fetch_openml`: downloading USPS and MNIST
- `scipy.optimize.linear_sum_assignment`: Hungarian matching in the accuracy metric
- `scipy.linalg.eigh`: the eigendecomposition of `L`
- `scipy.stats.ttest_ind`: the t-tests

Spectral rotation follows Yu and Shi, *Multiclass Spectral Clustering* (ICCV 2003), as cited by
the paper. Paper numbers in `src/table.py` are copied from Tables II–V of Yang et al. (2010).
