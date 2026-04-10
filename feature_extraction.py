# ============================================================
# PHASE 2: Feature Extraction — Frozen DINOv2 → PCA
# ============================================================
# INSTALL (if not already done):
#   !pip install timm -q

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────
PATCH_INDEX   = "/kaggle/working/patch_index.csv"
EMBED_OUT     = "/kaggle/working/embeddings_raw.npy"
PCA_EMBED_OUT = "/kaggle/working/embeddings_pca.npy"
PCA_MODEL_OUT = "/kaggle/working/pca_model.pkl"
PCA_DIMS      = 50       # reduce 768 → 50 (good for limited data)
BATCH_SIZE    = 64       # reduce to 32 if GPU OOM
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────

print(f"Using device: {DEVICE}")

# ── DATASET ─────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, patch_index_csv):
        self.df = pd.read_csv(patch_index_csv)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet stats (works well for DINOv2)
                std =[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")          # uint8 HxWx3
        return self.transform(img), idx      # return idx to preserve order

# ── MODEL — FROZEN DINOv2 ────────────────────────────────────
def load_dinov2():
    """
    Load DINOv2 ViT-S/14 (smallest, fastest — good for limited compute).
    Switch to vitb14 if you want richer features and have more GPU memory.
    """
    print("Loading DINOv2 (ViT-S/14)...")
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    except Exception:
        # Fallback: load via timm
        import timm
        print("  Hub load failed, falling back to timm...")
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                  pretrained=True, num_classes=0)
    model.eval()
    # Freeze ALL parameters — we never update weights
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(DEVICE)
    print(f" DINOv2 loaded and frozen on {DEVICE}")
    return model

# ── EXTRACT EMBEDDINGS ────────────────────────────────────────
def extract_embeddings(model, dataloader, n_patches):
    """
    Forward pass all patches through frozen ViT.
    Returns embeddings array of shape (n_patches, 384).
    384 = ViT-S embedding dim. ViT-B gives 768.
    """
    embeddings = np.zeros((n_patches, 384), dtype=np.float32)
    print(f"\nExtracting embeddings for {n_patches} patches...")

    with torch.no_grad():
        for batch_imgs, batch_idx in dataloader:
            batch_imgs = batch_imgs.to(DEVICE)
            feats      = model(batch_imgs)          # (B, 384)
            feats_np   = feats.cpu().numpy()
            for i, idx in enumerate(batch_idx.numpy()):
                embeddings[idx] = feats_np[i]

            done = batch_idx[-1].item() + 1
            if done % 1000 == 0 or done == n_patches:
                print(f"  {done}/{n_patches} patches processed...")

    print(f"  Raw embeddings shape: {embeddings.shape}")
    return embeddings

# ── PCA REDUCTION ─────────────────────────────────────────────
def fit_pca(embeddings, n_components=PCA_DIMS):
    """
    If dataset fits in memory, use standard PCA.
    For very large datasets (>100k patches), use IncrementalPCA.
    """
    n_patches = embeddings.shape[0]

    if n_patches > 80_000:
        print(f"\nLarge dataset ({n_patches} patches) — using IncrementalPCA...")
        pca = IncrementalPCA(n_components=n_components, batch_size=2048)
        pca.fit(embeddings)
    else:
        print(f"\nFitting PCA ({n_components} components) on {n_patches} patches...")
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(embeddings)

    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA fitted — {explained:.1f}% variance retained in {n_components} dims")
    return pca

# ── SANITY CHECK: does embedding vary across slides? ──────────
def sanity_check_embeddings(embeddings_pca, patch_index_csv):
    df = pd.read_csv(patch_index_csv)
    slides = df["slide_id"].unique()
    print(f"\n── Sanity Check: per-slide embedding mean ──")
    for s in slides:
        idx  = df[df["slide_id"] == s].index.tolist()
        mean = embeddings_pca[idx].mean(axis=0)[:3]   # show first 3 PCs
        print(f"  {s}: PC1={mean[0]:.3f}, PC2={mean[1]:.3f}, PC3={mean[2]:.3f}")
    print("  (Large differences between slides could indicate stain/batch effects)")

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load patch index
    df = pd.read_csv(PATCH_INDEX)
    print(f"Total patches to embed: {len(df)}")

    # Build dataset + dataloader
    dataset    = PatchDataset(PATCH_INDEX)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Load frozen model
    model = load_dinov2()

    # Extract raw embeddings
    embeddings_raw = extract_embeddings(model, dataloader, n_patches=len(df))
    np.save(EMBED_OUT, embeddings_raw)
    print(f"  Raw embeddings saved → {EMBED_OUT}")

    # Free GPU memory before PCA
    del model
    torch.cuda.empty_cache()

    # Fit PCA + transform
    pca             = fit_pca(embeddings_raw, n_components=PCA_DIMS)
    embeddings_pca  = pca.transform(embeddings_raw).astype(np.float32)

    np.save(PCA_EMBED_OUT, embeddings_pca)
    with open(PCA_MODEL_OUT, "wb") as f:
        pickle.dump(pca, f)

    print(f"\n  PCA embeddings saved → {PCA_EMBED_OUT}")
    print(f"  PCA model saved      → {PCA_MODEL_OUT}")

    # Sanity check
    sanity_check_embeddings(embeddings_pca, PATCH_INDEX)

    print(f"\n══ Phase 2 Complete ══")
    print(f"Embedding shape: {embeddings_pca.shape}  (n_patches × {PCA_DIMS})")
    print(f"Next step      : Run phase3_clustering.py")
