import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt

# ==========================================
# 0. setting 
# ==========================================
SCALE_M = 1000   # (pyramid resolution -> 10, 50, 70, 100, 200, 1000)
YEAR = 2022      # year parameter

input_file = os.path.join('data', f'DC_PYRAMID_{SCALE_M}m_moasic_2_{YEAR}.csv') 

# create result files
output_dir = f'result_{SCALE_M}m'
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. read data
# ==========================================
print(f"Reading file: {input_file} ...")
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Can not find file {input_file}")

df = pd.read_csv(input_file)

meta_cols = ['system:index', '.geo', 'src_count', 'norm_before', 'valid', 'x0', 'y0']
emb_cols = [col for col in df.columns if col not in meta_cols]

print(f"Extracting feature vectors, number of bands: {len(emb_cols)}...")
vectors = df[emb_cols].values.astype(np.float32)
N = vectors.shape[0]
print(f"Total number of {SCALE_M}m grid cells (N): {N:,}")

# ==========================================
# 2. initialize streaming global statistics
# ==========================================
bins = np.linspace(-1.0, 1.0, 101) 
global_hist_counts = np.zeros(100, dtype=np.int64)

total_pairs = 0
sum_sim = 0.0          
sum_sq_sim = 0.0       

chunk_size = 2000      
total_chunks = int(np.ceil(N / chunk_size))

# ==========================================
# 3. core streaming computation (Chunking Matrix Dot Product)
# ==========================================
print(f"\nStarting {SCALE_M}m chunked computation, total batches: {total_chunks}")
print("-" * 50)
start_time = time.time()

for chunk_idx, start in enumerate(range(0, N, chunk_size)):
    end = min(start + chunk_size, N)
    
    chunk_vectors = vectors[start:end]
    sim_block = np.dot(chunk_vectors, vectors.T)
    
    row_indices = np.arange(start, end)[:, None]  
    col_indices = np.arange(N)[None, :]           
    
    mask = col_indices > row_indices
    valid_sims = sim_block[mask]
    
    valid_sims = np.clip(valid_sims, -1.0, 1.0)
    
    counts, _ = np.histogram(valid_sims, bins=bins)
    global_hist_counts += counts
    
    total_pairs += valid_sims.size
    sum_sim += np.sum(valid_sims)
    sum_sq_sim += np.sum(valid_sims ** 2) 
    
    if (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == total_chunks:
        elapsed = time.time() - start_time
        print(f"  Completed batches: {chunk_idx + 1}/{total_chunks} | Accumulated pairs: {total_pairs:,} | Time elapsed: {elapsed:.1f}s")

print("-" * 50)
print(f"Computation finished! Total time: {time.time() - start_time:.2f} seconds.")

# ==========================================
# 4. compute final statistics and save
# ==========================================
print("\nGenerating statistical report and plots...")

exact_mean = sum_sim / total_pairs
exact_variance = (sum_sq_sim / total_pairs) - (exact_mean ** 2)
exact_std = np.sqrt(max(exact_variance, 0.0)) 

cumulative_counts = np.cumsum(global_hist_counts)
median_bin_idx = np.searchsorted(cumulative_counts, total_pairs / 2)
bin_centers = (bins[:-1] + bins[1:]) / 2.0
approx_median = bin_centers[median_bin_idx]

stats = {
    "resolution": f"{SCALE_M}m",
    "total_cells_N": int(N),
    "total_pairs_compared": int(total_pairs),
    "exact_mean": float(exact_mean),
    "exact_std_dev": float(exact_std),
    "approx_median": float(approx_median)
}

# save JSON
json_out = os.path.join(output_dir, f'summary_statistics_{SCALE_M}m.json')
with open(json_out, 'w') as f:
    json.dump(stats, f, indent=4)

# save CSV distribution
hist_df = pd.DataFrame({
    'bin_edge_left': bins[:-1],
    'bin_edge_right': bins[1:],
    'bin_center': bin_centers,
    'count': global_hist_counts
})
csv_out = os.path.join(output_dir, f'histogram_distribution_{SCALE_M}m.csv')
hist_df.to_csv(csv_out, index=False)

# ==========================================
# 5. visualization
# ==========================================
plt.figure(figsize=(12, 7), dpi=150)

plt.fill_between(bin_centers, global_hist_counts, step="mid", color='#d73027', alpha=0.7, edgecolor='#a50026')

plt.title(f"Cosine Similarity Distribution ({SCALE_M}m Resolution)\nWashington DC Embeddings ({YEAR})", fontsize=15, fontweight='bold')
plt.xlabel("Cosine Similarity", fontsize=12)
plt.ylabel("Frequency (Count)", fontsize=12)
plt.xlim(-1.0, 1.0)
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid(axis='y', linestyle='--', alpha=0.5)

stats_text = (f"Total {SCALE_M}m Cells: {N:,}\n"
              f"Pairs Computed: {total_pairs:,}\n"
              f"Mean Sim: {exact_mean:.4f}\n"
              f"Median Sim (est): {approx_median:.4f}\n"
              f"Std Dev: {exact_std:.4f}")

plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
output_img_path = os.path.join(output_dir, f'cosine_similarity_distribution_{SCALE_M}m.png')
plt.savefig(output_img_path)
plt.close()

print(f"All {SCALE_M}m files have been saved to ./{output_dir}/ directory.")