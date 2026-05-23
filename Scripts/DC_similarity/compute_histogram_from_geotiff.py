"""
Compute pairwise cosine similarity histogram from native 10m AEF GeoTIFF.

Output CSV matches the histogram_distribution_{scale}m.csv format exactly:
    bin_edge_left, bin_edge_right, bin_center, count

Dependencies
------------
    pip install rasterio numpy pandas tqdm

Usage
-----
    python compute_histogram_from_geotiff.py
"""

import os
import json
import time

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
INPUT_TIFF  = os.path.join('data', 'DC_AEF_10m_native_2022.tif')
OUTPUT_DIR  = 'result_10m'
SCALE_LABEL = '10m'
YEAR        = 2022

# Random subsample size.
# 50k pixels → ~1.25B pairs, ~2-4h on a modern CPU.
# Set to None to use all pixels (may take days).
# SAMPLE_N    = 50_000

# Right-side chunk size for the inner loop.
# Peak memory ≈ CHUNK_SIZE × 64 × 4 bytes ≈ 12 MB at CHUNK_SIZE=50_000.
CHUNK_SIZE  = 50_000

# Histogram bins — must match the pyramid scripts exactly
N_BINS = 100
BINS   = np.linspace(-1.0, 1.0, N_BINS + 1)   # 101 edges → 100 bins

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. Load GeoTIFF → (N, 64) unit-norm matrix
# ─────────────────────────────────────────
print(f"Reading {INPUT_TIFF} ...")
t0 = time.time()

with rasterio.open(INPUT_TIFF) as src:
    meta = {
        'crs':       str(src.crs),
        'transform': list(src.transform),
        'width':     src.width,
        'height':    src.height,
        'count':     src.count,
    }
    data = src.read()   # (bands, H, W)

n_bands, H, W = data.shape
print(f"  Bands={n_bands}, H={H}, W={W}, total pixels={H*W:,}  ({time.time()-t0:.1f}s)")

vectors_all = data.reshape(n_bands, -1).T.astype(np.float32)   # (N_total, 64)
del data

# Filter nodata: remove pixels that are all-zero, contain nan, or contain inf.
# rasterio fills masked areas with nan (not 0), so nan-check is essential.
valid_mask = (
    ~np.any(np.isnan(vectors_all), axis=1) &
    ~np.any(np.isinf(vectors_all), axis=1) &
     np.any(vectors_all != 0,      axis=1)
)
print(f"  Pixels removed by nodata filter: {(~valid_mask).sum():,} "
      f"({(~valid_mask).mean()*100:.1f}%)")
vectors = vectors_all[valid_mask]
del vectors_all
N_full = vectors.shape[0]
print(f"  Valid pixels after nodata filter: {N_full:,}")

# Re-normalize to unit sphere
norms    = np.linalg.norm(vectors, axis=1, keepdims=True)
norm_dev = np.abs(norms.squeeze() - 1.0)
print(f"  Norm deviation — max: {norm_dev.max():.6f}, mean: {norm_dev.mean():.6f}")
if norm_dev.max() > 1e-3:
    print("  WARNING: significant norm deviation, re-normalizing ...")
norms   = np.where(norms < 1e-9, 1.0, norms)
vectors = vectors / norms

# Final nan guard after normalization
nan_after = np.any(np.isnan(vectors), axis=1)
if nan_after.any():
    print(f"  WARNING: {nan_after.sum():,} nan vectors after normalization, removing ...")
    vectors = vectors[~nan_after]
    N_full  = vectors.shape[0]
print(f"  Clean vectors ready: {N_full:,}")

print(f"  Load + normalize done in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────
# 1.5 Random subsample
# ─────────────────────────────────────────
"""
if SAMPLE_N is not None and SAMPLE_N < N_full:
    rng     = np.random.default_rng(42)
    idx     = rng.choice(N_full, size=SAMPLE_N, replace=False)
    idx.sort()                  # sorted for cache-friendly access
    vectors = vectors[idx]
    N       = vectors.shape[0]
    print(f"  Subsampled {N_full:,} → {N:,} pixels (seed=42)")
else:
    N = N_full
    print(f"  Using all {N:,} pixels (no subsampling)")

expected_pairs = N * (N - 1) // 2
print(f"  Expected pairs: {expected_pairs:,}")

"""
"""
全量计算的cpu会用到这两行
"""
N = N_full
expected_pairs = N * (N - 1) // 2
print(f"  Using all {N:,} pixels, expected pairs: {expected_pairs:,}")
# ─────────────────────────────────────────
# 2. Streaming pairwise computation
#
# KEY FIX: use np.float64 numpy scalars for sum_sim and sum_sq_sim,
# NOT Python float. Accumulating ~1.25B float32 values with Python float
# causes catastrophic cancellation → nan.
# np.float64 accumulators keep full precision throughout.
#
# Additionally, each chunk's contribution is computed in float64
# via np.sum(..., dtype=np.float64) before adding to the accumulator,
# avoiding intermediate float32 precision loss.
# ─────────────────────────────────────────
global_hist = np.zeros(N_BINS, dtype=np.int64)
total_pairs = np.int64(0)
sum_sim     = np.float64(0.0)   # float64 accumulator, not Python float
sum_sq_sim  = np.float64(0.0)   # float64 accumulator, not Python float

print(f"\nStarting pairwise computation  (N={N:,}, CHUNK_SIZE={CHUNK_SIZE:,})")
print("-" * 60)
t1 = time.time()

pbar = tqdm(total=N, desc="pixels", unit="px")

for i in range(N):
    for j_start in range(i + 1, N, CHUNK_SIZE):
        j_end       = min(j_start + CHUNK_SIZE, N)
        right_chunk = vectors[j_start:j_end]              # (cs, 64)

        # Compute in float64 to avoid float32 precision loss in dot product
        sim_row = np.dot(
            vectors[i].astype(np.float64),
            right_chunk.T.astype(np.float64)
        )                                                 # (cs,) float64
        sim_row = np.clip(sim_row, -1.0, 1.0)

        counts, _    = np.histogram(sim_row, bins=BINS)
        global_hist += counts

        total_pairs += sim_row.size
        # np.sum with float64 dtype: safe even for large arrays
        sum_sim    += np.sum(sim_row, dtype=np.float64)
        sum_sq_sim += np.sum(sim_row ** 2, dtype=np.float64)

    pbar.update(1)

    if i > 0 and i % 5000 == 0:
        elapsed = time.time() - t1
        rate    = int(total_pairs) / elapsed / 1e6
        tqdm.write(f"  pixel {i:,}/{N:,} | pairs {int(total_pairs):,} | "
                   f"{rate:.1f}M pairs/s | {elapsed/60:.1f}min elapsed")

pbar.close()
print("-" * 60)
print(f"Computation done in {(time.time()-t1)/60:.1f} min  |  "
      f"total pairs: {int(total_pairs):,}")

#debug
print(f"DEBUG sum_sim     = {sum_sim},  type={type(sum_sim)},  isnan={np.isnan(sum_sim)}")
print(f"DEBUG sum_sq_sim  = {sum_sq_sim}, type={type(sum_sq_sim)}, isnan={np.isnan(sum_sq_sim)}")
print(f"DEBUG total_pairs = {total_pairs}")
# ─────────────────────────────────────────
# 3. Summary statistics
# ─────────────────────────────────────────
exact_mean     = float(sum_sim) / int(total_pairs)
exact_variance = float(sum_sq_sim) / int(total_pairs) - exact_mean ** 2
exact_std      = np.sqrt(max(exact_variance, 0.0))

cumulative    = np.cumsum(global_hist)
median_idx    = np.searchsorted(cumulative, int(total_pairs) / 2)
bin_centers   = (BINS[:-1] + BINS[1:]) / 2.0
approx_median = float(bin_centers[min(median_idx, N_BINS - 1)])

# Sanity check: warn if statistics look wrong
if np.isnan(exact_mean) or np.isnan(exact_std):
    print("WARNING: nan detected in statistics — check accumulator types")
if not (-1.0 <= exact_mean <= 1.0):
    print(f"WARNING: mean {exact_mean:.6f} out of [-1, 1] range")

stats = {
    "resolution":           SCALE_LABEL,
    "year":                 YEAR,
    "total_pixels_N_full":  int(N_full),
    "sample_n":             int(N),
    "total_pairs_compared": int(total_pairs),
    "exact_mean":           float(exact_mean),
    "exact_std_dev":        float(exact_std),
    "approx_median":        approx_median,
    "source_tiff":          INPUT_TIFF,
    "geotiff_meta":         meta,
    "note_pvalue": (
        "p-values from KS tests are not interpretable due to spatial "
        "autocorrelation between pixel pairs. Use KS D statistic only."
    )
}

# ─────────────────────────────────────────
# 4. Save outputs
# ─────────────────────────────────────────
hist_df = pd.DataFrame({
    'bin_edge_left':  BINS[:-1],
    'bin_edge_right': BINS[1:],
    'bin_center':     bin_centers,
    'count':          global_hist,
})
csv_path = os.path.join(OUTPUT_DIR, f'histogram_distribution_{SCALE_LABEL}.csv')
hist_df.to_csv(csv_path, index=False)
print(f"Saved histogram → {csv_path}")

json_path = os.path.join(OUTPUT_DIR, f'summary_statistics_{SCALE_LABEL}.json')
with open(json_path, 'w') as f:
    json.dump(stats, f, indent=4)
print(f"Saved stats     → {json_path}")

print(f"""
Summary ({SCALE_LABEL} native pixels, sample={N:,}/{N_full:,})
  Total pairs   : {int(total_pairs):,}
  Mean sim      : {exact_mean:.6f}
  Std dev       : {exact_std:.6f}
  Median (est)  : {approx_median:.6f}

Feed {csv_path} into ks_test_distributions.py for comparison.
""")