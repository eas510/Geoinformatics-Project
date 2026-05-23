"""
KS test between multiple cosine-similarity distributions
stored as histogram CSVs (bin_edge_left, bin_edge_right, bin_center, count).

Outputs
-------
- ks_matrix.csv          : pairwise KS statistic D
- pvalue_matrix.csv      : pairwise p-value
- ks_heatmap.png         : annotated heatmap of D
- cdf_overlay.png        : overlaid empirical CDFs
- ks_results_detail.json : full result record
"""

import os
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import ks_2samp

# ─────────────────────────────────────────
# CONFIG  — edit these paths and labels
# ─────────────────────────────────────────
INPUT_FILES = [
    os.path.join('data', 'histogram_distribution_10m.csv'),
    os.path.join('data', 'histogram_distribution_50m.csv'),
    os.path.join('data', 'histogram_distribution_100m.csv'),
    os.path.join('data', 'histogram_distribution_300m.csv'),
    os.path.join('data', 'histogram_distribution_1000m.csv'),
    # add more as needed
]

LABELS = [
    '10m',
    '50m',
    '100m',
    '300m',
    '1000m'
]

OUTPUT_DIR = 'ks_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. Load histogram CSVs → empirical CDFs
# ─────────────────────────────────────────

def load_ecdf(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a histogram CSV and return (bin_centers, cdf).

    The CDF is built from the *right* edges so that
      CDF[i] = P(X <= bin_edge_right[i])
    This matches the convention used by scipy.stats.ks_2samp
    when it operates on the sorted-sample representation.

    Returns
    -------
    edges_right : (100,) array of right bin edges
    cdf         : (100,) array, last value is exactly 1.0
    """
    df = pd.read_csv(filepath)

    required = {'bin_edge_left', 'bin_edge_right', 'bin_center', 'count'}
    if not required.issubset(df.columns):
        raise ValueError(f"{filepath} is missing columns: {required - set(df.columns)}")

    counts = df['count'].values.astype(np.float64)
    total  = counts.sum()
    if total == 0:
        raise ValueError(f"{filepath} has all-zero counts.")

    edges_right = df['bin_edge_right'].values
    cdf = np.cumsum(counts) / total
    cdf[-1] = 1.0          # enforce exact 1.0 at the last bin

    return edges_right, cdf


def ecdf_to_pseudo_samples(edges_right: np.ndarray,
                            cdf: np.ndarray,
                            n_samples: int = 10_000) -> np.ndarray:
    """
    Convert a CDF defined on bin right-edges to a sorted pseudo-sample
    array of length n_samples via inverse CDF (quantile) interpolation.

    This lets us pass the distribution to scipy.stats.ks_2samp, which
    internally works on sorted samples and computes the exact KS statistic.
    The interpolation error is < bin_width / 2 = 0.01, well within the
    precision of 0.02-wide bins.
    """
    # prepend (−1.0, 0.0) so interpolation covers the full [-1, 1] domain
    x = np.concatenate([[-1.0], edges_right])
    y = np.concatenate([[0.0],  cdf])

    quantiles = np.linspace(0.0, 1.0, n_samples, endpoint=False) + 0.5 / n_samples
    samples   = np.interp(quantiles, y, x)
    return np.sort(samples)


print("Loading histogram files ...")
ecdfs   = {}   # label -> (edges_right, cdf)
samples = {}   # label -> pseudo_samples

for fpath, label in zip(INPUT_FILES, LABELS):
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File not found: {fpath}")
    edges_right, cdf = load_ecdf(fpath)
    ecdfs[label]     = (edges_right, cdf)
    samples[label]   = ecdf_to_pseudo_samples(edges_right, cdf)
    print(f"  Loaded '{label}' from {fpath}")

# ─────────────────────────────────────────
# 2. Pairwise two-sample KS test
# ─────────────────────────────────────────
# scipy.stats.ks_2samp(a, b) returns:
#   statistic : D = max|F_a(x) - F_b(x)|   (in [0, 1])
#   pvalue    : two-sided p-value under H0 (same distribution)
#
# Interpretation of D:
#   D < 0.05  → distributions nearly identical
#   D < 0.10  → very similar
#   D < 0.20  → moderate difference
#   D >= 0.20 → substantial difference

n      = len(LABELS)
D_mat  = np.zeros((n, n))
pv_mat = np.ones((n, n))

results_detail = []

print("\nRunning pairwise KS tests ...")
for (i, la), (j, lb) in itertools.combinations(enumerate(LABELS), 2):
    stat, pval = ks_2samp(samples[la], samples[lb], method='exact')
    D_mat[i, j]  = D_mat[j, i]  = stat
    pv_mat[i, j] = pv_mat[j, i] = pval

    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    print(f"  {la} vs {lb}: D = {stat:.6f}, p = {pval:.4e}  {sig}")

    results_detail.append({
        "pair": f"{la}_vs_{lb}",
        "label_a": la,
        "label_b": lb,
        "ks_statistic_D": round(float(stat), 8),
        "p_value": float(pval),
        "significant_0.05": bool(pval < 0.05),
        "interpretation": (
            "nearly identical" if stat < 0.05 else
            "very similar"     if stat < 0.10 else
            "moderate diff"    if stat < 0.20 else
            "substantial diff"
        )
    })

# ─────────────────────────────────────────
# 3. Save matrices
# ─────────────────────────────────────────
D_df  = pd.DataFrame(D_mat,  index=LABELS, columns=LABELS)
pv_df = pd.DataFrame(pv_mat, index=LABELS, columns=LABELS)

ks_csv_path = os.path.join(OUTPUT_DIR, 'ks_matrix.csv')
pv_csv_path = os.path.join(OUTPUT_DIR, 'pvalue_matrix.csv')
D_df.to_csv(ks_csv_path)
pv_df.to_csv(pv_csv_path)

json_path = os.path.join(OUTPUT_DIR, 'ks_results_detail.json')
with open(json_path, 'w') as f:
    json.dump(results_detail, f, indent=4)

print(f"\nSaved matrices → {ks_csv_path}, {pv_csv_path}")
print(f"Saved detail   → {json_path}")

# ─────────────────────────────────────────
# 4. Heatmap of KS statistic D
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(5, n + 1), max(4, n)), dpi=150)

im = ax.imshow(D_mat, vmin=0, vmax=0.3, cmap='RdYlGn_r', aspect='auto')
plt.colorbar(im, ax=ax, label='KS statistic D  (0 = identical, 1 = maximally different)')

ax.set_xticks(range(n)); ax.set_xticklabels(LABELS, rotation=30, ha='right')
ax.set_yticks(range(n)); ax.set_yticklabels(LABELS)

for i in range(n):
    for j in range(n):
        val  = D_mat[i, j]
        text = "—" if i == j else f"{val:.4f}"
        color = 'white' if val > 0.18 else 'black'
        ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

ax.set_title('Pairwise KS statistic D\n(Cosine Similarity Distributions)', fontsize=12)
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, 'ks_heatmap.png')
plt.savefig(heatmap_path)
plt.close()
print(f"Saved heatmap  → {heatmap_path}")

# ─────────────────────────────────────────
# 4.5 Heatmap of p-values
# ─────────────────────────────────────────
# Note: p-values are not statistically interpretable here due to
# spatial autocorrelation between pairs. Shown for completeness only.

fig, ax = plt.subplots(figsize=(max(5, n + 1), max(4, n)), dpi=150)

# Log-transform p-values for visualization: log10(p) in [-∞, 0]
# Clip at 1e-300 to avoid -inf; diagonal stays at 1.0 → log10 = 0
pv_display = np.where(np.eye(n, dtype=bool), 1.0, pv_mat)
log_pv     = np.log10(np.clip(pv_display, 1e-300, 1.0))

# Color scale: 0 (p=1, white/green) → -300 (p≈0, red)
# vmax=0 always; vmin set to the most extreme value in the matrix
off_diag   = log_pv[~np.eye(n, dtype=bool)]
vmin_val   = min(off_diag.min(), -1.0)   # at least one decade

im2 = ax.imshow(log_pv, vmin=vmin_val, vmax=0,
                cmap='RdYlGn', aspect='auto')

cbar = plt.colorbar(im2, ax=ax)
cbar.set_label('log₁₀(p-value)  [0 = p=1,  more negative = smaller p]')

# Tick marks at interpretable thresholds
cbar.set_ticks([0, -1, -2, -3, -5, -10, -50])
cbar.set_ticklabels(['1', '0.1', '0.01', '0.001', '1e-5', '1e-10', '1e-50'])

ax.set_xticks(range(n)); ax.set_xticklabels(LABELS, rotation=30, ha='right')
ax.set_yticks(range(n)); ax.set_yticklabels(LABELS)

for i in range(n):
    for j in range(n):
        if i == j:
            text  = '—'
            color = 'black'
        else:
            pv  = pv_mat[i, j]
            # Display as scientific notation or significance stars
            if pv < 1e-10:
                text = f'<1e-10'
            elif pv < 0.001:
                text = f'{pv:.1e}'
            else:
                text = f'{pv:.3f}'
            color = 'white' if log_pv[i, j] < vmin_val * 0.6 else 'black'
        ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

ax.set_title(
    'Pairwise p-values (KS test)\n'
    '(log scale — not interpretable due to spatial autocorrelation)',
    fontsize=11
)
plt.tight_layout()

pv_heatmap_path = os.path.join(OUTPUT_DIR, 'pvalue_heatmap.png')
plt.savefig(pv_heatmap_path)
plt.close()
print(f"Saved p-value heatmap → {pv_heatmap_path}")

# ─────────────────────────────────────────
# 5. CDF overlay plot
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

colors = plt.cm.tab10(np.linspace(0, 0.9, n))

for label, color in zip(LABELS, colors):
    edges_right, cdf = ecdfs[label]
    # prepend the left edge of the first bin for a step that starts at −1
    x_plot = np.concatenate([[-1.0], edges_right])
    y_plot = np.concatenate([[0.0],  cdf])
    ax.step(x_plot, y_plot, where='post', label=label, color=color, linewidth=1.5)

ax.set_xlabel('Cosine Similarity', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('Empirical CDFs — Cosine Similarity Distributions', fontsize=12)
ax.set_xlim(-0.1, 1.0)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.grid(axis='both', linestyle='--', alpha=0.4)
ax.legend(fontsize=10)
plt.tight_layout()

cdf_path = os.path.join(OUTPUT_DIR, 'cdf_overlay.png')
plt.savefig(cdf_path)
plt.close()
print(f"Saved CDF plot → {cdf_path}")

print("\nAll done.")