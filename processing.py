# ============================================================
# CELL 0 — RUN THIS FIRST IN YOUR KAGGLE NOTEBOOK
# Copy-paste this entire cell and run it before anything else
# ============================================================

# System dependencies for OpenSlide (.mrxs support)
import subprocess
subprocess.run(["apt-get", "install", "-y", "-q",
                "openslide-tools", "libopenslide-dev"], check=True)

# Python packages
subprocess.run(["pip", "install", "-q",
                "openslide-python",   # read .mrxs files
                "torchstain",         # Macenko stain normalization
                "timm",               # fallback ViT model loader
                "hdbscan",            # density clustering
                "umap-learn",         # UMAP visualization
                ], check=True)

print(" All dependencies installed")

# ── Verify OpenSlide works ────────────────────────────────────
import openslide
print(f" OpenSlide version: {openslide.__version__}")

# ── Verify GPU ────────────────────────────────────────────────
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── List your .mrxs files ─────────────────────────────────────
import os
from pathlib import Path

# ← CHANGE THIS to your actual Kaggle dataset path
SLIDE_DIR = "/kaggle/input/datasets/yepp26/check1/041"

slides = list(Path(SLIDE_DIR).rglob("*.mrxs"))
print(f"\n✓ Found {len(slides)} .mrxs slides:")
for s in slides:
    size_mb = s.stat().st_size / 1e6
    print(f"  {s.name}  ({size_mb:.0f} MB)")

# ── Quick open test on first slide ────────────────────────────
if slides:
    test_slide = openslide.OpenSlide(str(slides[0]))
    print(f"\n Test open: {slides[0].name}")
    print(f"  Levels    : {test_slide.level_count}")
    print(f"  Level 0   : {test_slide.level_dimensions[0]}")
    if test_slide.level_count > 1:
        print(f"  Level 1   : {test_slide.level_dimensions[1]}")
    if test_slide.level_count > 2:
        print(f"  Level 2   : {test_slide.level_dimensions[2]}")
    test_slide.close()
    print("\n Slide readable — you're good to go!")
else:
    print("\n No slides found. Check your SLIDE_DIR path.")

# ── Execution order reminder ──────────────────────────────────
print("""
══════════════════════════════════════════════
  EXECUTION ORDER:
  Cell 0  → This setup cell          (run once)
  Phase 1 → phase1_patching.py       (extract patches)
  Phase 2 → phase2_feature_extraction.py  (DINOv2 embeddings)
  Phase 3 → phase3_clustering.py     (K-Means + HDBSCAN)
  Phase 4 → phase4_spatial_maps.py   (maps + validation)
  Phase 5 → phase5_visualization.py  (UMAP + thumbnails)

  ⚠ IMPORTANT: After Phase 2, your embeddings are saved to
  /kaggle/working/embeddings_pca.npy — if session restarts,
  you can skip Phases 1 & 2 and load from this file directly.
══════════════════════════════════════════════
""")
