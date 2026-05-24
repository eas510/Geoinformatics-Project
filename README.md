# Geoinformatics-Project

# AEF Multi-Scale Aggregation Assessment

This repository contains a reproducible pipeline for evaluating multi-scale spatial aggregation of AlphaEarth Foundations (AEF) embeddings.

The goal of this project is to test how much information is preserved when native 10 m AEF embeddings are aggregated to coarser spatial resolutions. The assessment is based on pair-wise cosine similarity distributions and the two-sample Kolmogorov–Smirnov (K-S) test.

## Workflow

The pipeline has two main parts:

1. **Google Earth Engine processing**

   The GEE script loads the AEF embedding asset from Google Earth Engine and aggregates the embeddings to multiple target resolutions.

   For each target resolution, the downsampled embeddings are exported as `.csv` files. The native 10 m data is too large to export as a table, so it is saved as a GeoTIFF instead.

2. **Python similarity analysis**

   The scripts in the `DC_similarity` folder compute cosine similarity distributions from the exported data.

   - `compute_histogram_from_geotiff.py` computes the cosine similarity distribution for the native 10 m GeoTIFF.
   - `sim_computation.py` computes cosine similarity distributions for the downsampled `.csv` files at coarser resolutions.
   - The computed results are saved as `.json` files.
   - `plot_histogram.py` renders the histogram results for visualization.

## Method

AEF embeddings are high-dimensional feature vectors. Since the embedding dimensions do not have direct physical meanings, traditional band-wise remote sensing accuracy metrics are not suitable here.

Instead, this project compares the distribution of pair-wise cosine similarities across spatial resolutions. If a coarser resolution preserves the original embedding structure well, its cosine similarity distribution should remain close to the native 10 m distribution.

The K-S test is then used to quantify the difference between distributions from different resolutions.