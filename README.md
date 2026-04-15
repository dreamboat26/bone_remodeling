# Bone Remodeling Analysis Self Supervised Pipeline

End‑to‑end pipeline for self-supervised analysis of H&E whole‑slide images (`.mrxs` format). The pipeline extracts tissue patches, computes deep features with DINOv2, clusters patches to discover histological states, and generates spatial maps and visualisations.

## Overview

The pipeline consists of five sequential phases:

| Phase | Description |
|-------|-------------|
| **Phase 1** | Core detection and patch extraction with stain normalisation |
| **Phase 2** | Feature extraction using frozen DINOv2 (ViT‑S/14) + PCA |
| **Phase 3** | Unsupervised clustering with K‑Means and HDBSCAN |
| **Phase 4** | Spatial mapping, composition analysis, and validation |
| **Phase 5** | Embedding visualisation (PCA, UMAP) and thumbnail galleries |

All processing is CPU‑friendly for stain operations and memory‑aware to avoid GPU out‑of‑memory errors.

---

## Requirements

- Python 3.8 or higher
- CUDA‑capable GPU (optional – falls back to CPU)
- OpenSlide system libraries

## Installation

Run the following commands **once** in your environment (Kaggle/Colab/local):

```bash
# System dependencies for .mrxs support
apt-get install -y openslide-tools libopenslide-dev -q

# Python packages
pip install openslide-python torchstain opencv-python-headless timm hdbscan umap-learn -q
```

## Execution Order

Run the scripts in the following order. Each phase saves intermediate results and is required for the next step.

| Step | Script | Command | What it does |
|------|--------|---------|--------------|
| 1 | `phase1_patching.py` | `python phase1_patching.py` | Detects TMA cores, extracts patches, applies stain normalisation. Outputs patches and `patch_index.csv`. |
| 2 | `phase2_feature_extraction.py` | `python phase2_feature_extraction.py` | Extracts DINOv2 features and reduces with PCA. Outputs `embeddings_pca.npy`. |
| 3 | `phase3_clustering.py` | `python phase3_clustering.py` | Clusters patches with K‑Means and HDBSCAN. Outputs `cluster_assignments.csv`. |
| 4 | `phase4_spatial_maps.py` | `python phase4_spatial_maps.py` | Projects clusters onto slide coordinates, generates spatial maps and validation plots. |
| 5 | `phase5_visualization.py` | `python phase5_visualization.py` | Creates PCA/UMAP visualisations and thumbnail galleries. |

## Visualizations

For visualizations refer to the folder named Visualizations, we have put all visualizations for Case 009 and Case 218 for better understanding. 
