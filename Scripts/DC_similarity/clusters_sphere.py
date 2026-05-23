import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import time

# ==========================================
# Input & Output
# ==========================================
input_file = 'data/DC_PYRAMID_1000m_moasic_2_2022.csv'  
output_dir = 'result_1000m'
n_clusters = 14  # classification number
output_file = os.path.join(output_dir, f'DC_1000m_Clustered_k{n_clusters}_sphere.csv')

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# read data and background pixels filtering
# ==========================================
print(f"Reading file: {input_file} ...")
start_total = time.time()
df_raw = pd.read_csv(input_file)

# use 'valid' field to filter the empty pixels
if 'valid' in df_raw.columns:
    df = df_raw[df_raw['valid'] == True].copy()
else:
    df = df_raw.copy()

meta_cols = ['system:index', '.geo', 'src_count', 'norm_before', 'valid', 'x0', 'y0']
emb_cols = [col for col in df.columns if col not in meta_cols]

# generating ndarry for kmeans clustering
X = df[emb_cols].values.astype(np.float32)
N, D = X.shape
# print(f"有效城市网格: {N:,} | 特征维度: {D}")

# ==========================================
# Spherical K-Means(based on cosine distance)
# ==========================================
print(f"\nSpherical K-Means (k={n_clusters})...")
start_cluster = time.time()

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X)

print(f"Kmeans finished {time.time() - start_cluster:.1f}s")

df['cluster'] = cluster_labels
final_cols = ['system:index', 'x0', 'y0', 'cluster']
print("\nCalculating silhouette score...")
start_eval = time.time()

# Silhouette score is expensive for large N, sample if needed
MAX_SAMPLE = 10000
if N > MAX_SAMPLE:
    print(f"Dataset too large ({N:,} grids), sampling {MAX_SAMPLE:,} for silhouette evaluation...")
    sample_idx = np.random.choice(N, MAX_SAMPLE, replace=False)
    X_sample = X[sample_idx]
    labels_sample = cluster_labels[sample_idx]
else:
    X_sample = X
    labels_sample = cluster_labels

score = silhouette_score(X_sample, labels_sample, metric='cosine')
print(f"Silhouette Score (k={n_clusters}): {score:.4f}  |  Time: {time.time() - start_eval:.1f}s")
print(f"  > Score interpretation: [-1, 1], higher is better")
print(f"  > > 0.5: strong structure | 0.25~0.5: moderate | < 0.25: weak")

print(f"Saving result to {output_file} ...")
df[final_cols].to_csv(output_file, index=False)
print(f"Total time consumption: {time.time() - start_total:.2f}s。")