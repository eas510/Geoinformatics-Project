"""
Plot cosine similarity histogram from a histogram_distribution_{scale}m.csv file,
matching the style of the existing 50m plot.

Multiple color themes available — set THEME below.

Usage
-----
    python plot_histogram.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
INPUT_CSV   = os.path.join('result_10m', 'histogram_distribution_10m.csv')
STATS_JSON  = os.path.join('result_10m', 'summary_statistics_10m.json')
OUTPUT_DIR  = 'result_10m'
SCALE_LABEL = '10m'
YEAR        = 2022
REGION      = 'Washington DC'

# ── Color theme
# Options:
#   'red'       original 50m style (crimson-red)
#   'blue'      steel blue
#   'teal'      muted teal / seafoam
#   'purple'    dusty purple
#   'olive'     olive / warm green
#   'slate'     cool slate gray
#   'amber'     warm amber / ochre
#   'rose'      muted rose / mauve
THEME = 'purple'

THEMES = {
    'red':    {'fill': '#d73027', 'edge': '#a50026', 'alpha': 0.70},
    'blue':   {'fill': '#2166ac', 'edge': '#053061', 'alpha': 0.70},
    'teal':   {'fill': '#1d9e75', 'edge': '#0a5e43', 'alpha': 0.68},
    'purple': {'fill': '#7b3fa0', 'edge': '#4a1a6b', 'alpha': 0.68},
    'olive':  {'fill': '#5a7a2b', 'edge': '#324815', 'alpha': 0.70},
    'slate':  {'fill': '#4d6b8a', 'edge': '#27404f', 'alpha': 0.70},
    'amber':  {'fill': '#c07b1a', 'edge': '#7a4a05', 'alpha': 0.72},
    'rose':   {'fill': '#c2556e', 'edge': '#7d2840', 'alpha': 0.70},
}

if THEME not in THEMES:
    raise ValueError(f"Unknown THEME '{THEME}'. Choose from: {list(THEMES.keys())}")

color = THEMES[THEME]

# ─────────────────────────────────────────
# 1. Load histogram CSV
# ─────────────────────────────────────────
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"File not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
bin_centers = df['bin_center'].values
counts      = df['count'].values.astype(np.float64)

# ─────────────────────────────────────────
# 2. Load summary statistics
# ─────────────────────────────────────────
stats = {}
if os.path.exists(STATS_JSON):
    with open(STATS_JSON) as f:
        stats = json.load(f)

total_cells = stats.get('sample_n',             stats.get('total_cells_N', 0))
total_pairs = stats.get('total_pairs_compared', 0)
mean_sim    = stats.get('exact_mean',           np.nan)
median_sim  = stats.get('approx_median',        np.nan)
std_dev     = stats.get('exact_std_dev',        np.nan)

# ─────────────────────────────────────────
# 3. Plot
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

ax.fill_between(
    bin_centers,
    counts,
    step='mid',
    color=color['fill'],
    alpha=color['alpha'],
    edgecolor=color['edge'],
    linewidth=0.6,
)

# ── Titles and labels (matching 50m style exactly)
ax.set_title(
    f'Cosine Similarity Distribution ({SCALE_LABEL} Resolution)\n'
    f'{REGION} Embeddings ({YEAR})',
    fontsize=15,
    fontweight='bold',
)
ax.set_xlabel('Cosine Similarity', fontsize=12)
ax.set_ylabel('Frequency (Count)',  fontsize=12)

ax.set_xlim(-1.0, 1.0)
ax.set_ylim(bottom=0)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.grid(axis='y', linestyle='--', alpha=0.5)

# ── Stats annotation box
stats_text = (
    f'Total {SCALE_LABEL} Cells: {total_cells:,}\n'
    f'Pairs Computed: {total_pairs:,}\n'
    f'Mean Sim: {mean_sim:.4f}\n'
    f'Median Sim (est): {median_sim:.4f}\n'
    f'Std Dev: {std_dev:.4f}'
)

ax.text(
    0.05, 0.95,
    stats_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment='top',
    family='monospace',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        alpha=0.8,
        edgecolor='gray',
    ),
)

plt.tight_layout()

# ─────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(
    OUTPUT_DIR,
    f'cosine_similarity_distribution_{SCALE_LABEL}_{THEME}.png'
)
plt.savefig(out_path)
plt.close()
print(f"Saved → {out_path}")