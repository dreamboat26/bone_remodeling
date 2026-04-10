# ============================================================
# PHASE 5: Embedding Visualization (UMAP/PCA) 
# ============================================================
# INSTALL:
#   !pip install umap-learn -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────
PCA_EMBED_PATH = "/kaggle/working/embeddings_pca.npy"
CLUSTER_CSV    = "/kaggle/working/clustering/cluster_assignments.csv"
RESULTS_DIR    = "/kaggle/working/visualizations"
N_VIZ_SAMPLES  = 8000    # subsample for UMAP speed (all patches if < this)
RANDOM_STATE   = 42
# ────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

def subsample(embeddings, df, n=N_VIZ_SAMPLES):
    if len(embeddings) > n:
        idx = np.random.choice(len(embeddings), n, replace=False)
        return embeddings[idx], df.iloc[idx].reset_index(drop=True)
    return embeddings, df.reset_index(drop=True)

# ── 2D PCA PLOT (fast, always works) ─────────────────────────
def plot_pca2d(embeddings, df, color_col, title_suffix=""):
    from sklearn.decomposition import PCA
    pca2  = PCA(n_components=2, random_state=RANDOM_STATE)
    proj  = pca2.fit_transform(embeddings)

    labels     = df[color_col].values
    unique_lbs = sorted(set(labels))
    cmap       = plt.get_cmap("tab10", len(unique_lbs))
    color_map  = {l: cmap(i) for i, l in enumerate(unique_lbs)}
    colors     = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=4, alpha=0.5, linewidths=0)

    legend_handles = []
    for l in unique_lbs:
        name = "Anomaly" if l == -1 else (
            l if isinstance(l, str) else f"State {l}"
        )
        legend_handles.append(mpatches.Patch(color=color_map[l], label=name))
    ax.legend(handles=legend_handles, loc="best", fontsize=8, markerscale=2)

    exp_var = pca2.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}% var)")
    ax.set_title(f"PCA Embedding — colored by {color_col}{title_suffix}")
    ax.grid(alpha=0.2)

    out = f"{RESULTS_DIR}/pca2d_{color_col}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Saved: {out}")

# ── UMAP PLOT (richer structure, takes ~1-2 min) ──────────────
def plot_umap(embeddings, df, color_col, title_suffix=""):
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed. Run: !pip install umap-learn -q")
        return

    print(f"  Running UMAP on {len(embeddings)} points...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="euclidean", random_state=RANDOM_STATE, verbose=False)
    proj    = reducer.fit_transform(embeddings)

    labels     = df[color_col].values
    unique_lbs = sorted(set(labels))
    cmap       = plt.get_cmap("tab10", len(unique_lbs))
    color_map  = {l: cmap(i) for i, l in enumerate(unique_lbs)}
    colors     = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=4, alpha=0.5, linewidths=0)

    legend_handles = []
    for l in unique_lbs:
        name = "Anomaly" if l == -1 else (
            l if isinstance(l, str) else f"State {l}"
        )
        legend_handles.append(mpatches.Patch(color=color_map[l], label=name))
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP Embedding — colored by {color_col}{title_suffix}")
    ax.set_xticks([]); ax.set_yticks([])

    out = f"{RESULTS_DIR}/umap_{color_col}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Saved: {out}")
    return proj

# ── PATCH THUMBNAILS PER CLUSTER ─────────────────────────────
def show_cluster_thumbnails(df, cluster_col="cluster_kmeans", n_per_cluster=6):
    """
    Show sample patches from each cluster.
    This is how you biologically interpret what each 'state' looks like.
    """
    print(f"\n── Cluster Thumbnails ({cluster_col}) ──")
    cluster_ids = sorted(df[cluster_col].unique())
    n_clusters  = len(cluster_ids)

    fig, axes = plt.subplots(n_clusters, n_per_cluster,
                             figsize=(n_per_cluster * 2, n_clusters * 2))
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    for row_i, c in enumerate(cluster_ids):
        sub     = df[df[cluster_col] == c]
        samples = sub.sample(min(n_per_cluster, len(sub)),
                             random_state=RANDOM_STATE)
        label   = "Anomaly" if c == -1 else f"State {c}"

        for col_i in range(n_per_cluster):
            ax = axes[row_i, col_i]
            ax.axis("off")
            if col_i < len(samples):
                try:
                    patch = np.load(samples.iloc[col_i]["path"])
                    ax.imshow(patch)
                    if col_i == 0:
                        ax.set_title(label, fontsize=8, fontweight="bold",
                                     loc="left", pad=2)
                except Exception:
                    ax.set_facecolor("#eee")

    plt.suptitle(f"Sample Patches per Cluster ({cluster_col})",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/thumbnails_{cluster_col}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out}")

# ── FINAL SUMMARY REPORT ─────────────────────────────────────
def print_final_report(df):
    print("\n" + "═"*60)
    print("  PIPELINE SUMMARY REPORT")
    print("═"*60)
    print(f"\n  Slides processed   : {df['slide_id'].nunique()}")
    print(f"  Total patches      : {len(df)}")
    print(f"\n  K-Means clusters   : {df['cluster_kmeans'].nunique()}")
    print(f"  HDBSCAN clusters   : {df['cluster_hdbscan'].nunique() - (1 if -1 in df['cluster_hdbscan'].values else 0)}")
    print(f"  Anomaly patches    : {df['is_anomaly'].sum()} "
          f"({100*df['is_anomaly'].mean():.1f}%)")

    print("\n  Per-slide stats:")
    for sid in df["slide_id"].unique():
        sub = df[df["slide_id"] == sid]
        dominant = sub["cluster_kmeans"].mode()[0]
        anom_pct = 100 * sub["is_anomaly"].mean()
        print(f"    {sid}")
        print(f"      Patches      : {len(sub)}")
        print(f"      Dominant state: State {dominant}")
        print(f"      Anomaly %    : {anom_pct:.1f}%")

    print("\n  Output files:")
    for root, _, files in os.walk("/kaggle/working"):
        for f in files:
            if f.endswith((".png", ".csv", ".npy", ".pkl")):
                print(f"    /kaggle/working/{os.path.relpath(os.path.join(root, f), '/kaggle/working')}")

    print("\n  Next scientific steps:")
    print("    1. Look at thumbnails — what tissue does each state show?")
    print("    2. Check spatial maps — do states form coherent regions?")
    print("    3. Check anomaly slides — do they match the distinctive patient?")
    print("    4. Run with metadata CSV to check age/sex correlations.")
    print("═"*60)

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    embeddings = np.load(PCA_EMBED_PATH)
    df         = pd.read_csv(CLUSTER_CSV)
    print(f"Loaded {len(embeddings)} embeddings, {len(df)} patch records")

    # Subsample for visualization speed
    emb_sub, df_sub = subsample(embeddings, df, n=N_VIZ_SAMPLES)
    print(f"Using {len(emb_sub)} samples for UMAP/PCA plots")

    # PCA 2D — color by K-means cluster
    print("\n── PCA 2D Plots ──")
    plot_pca2d(emb_sub, df_sub, color_col="cluster_kmeans")
    plot_pca2d(emb_sub, df_sub, color_col="cluster_hdbscan")
    plot_pca2d(emb_sub, df_sub, color_col="slide_id",
               title_suffix=" (sanity: should mix slides)")

    # UMAP — richer structure
    print("\n── UMAP Plots ──")
    plot_umap(emb_sub, df_sub, color_col="cluster_kmeans")
    plot_umap(emb_sub, df_sub, color_col="cluster_hdbscan")
    plot_umap(emb_sub, df_sub, color_col="slide_id")

    # Thumbnail gallery — KEY for biological interpretation
    print("\n── Thumbnail Gallery ──")
    show_cluster_thumbnails(df, cluster_col="cluster_kmeans")
    show_cluster_thumbnails(df, cluster_col="cluster_hdbscan")

    # Final summary
    print_final_report(df)

    print(f"\n══ Pipeline Complete ══")
    print(f"All outputs in /kaggle/working/")
