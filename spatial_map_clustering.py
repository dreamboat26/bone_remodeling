# ============================================================
# PHASE 3: Unsupervised Clustering — HDBSCAN + K-means
# ============================================================
# INSTALL (if not already done):
#   !pip install hdbscan -q

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────
PCA_EMBED_PATH = "/kaggle/working/embeddings_pca.npy"
PATCH_INDEX    = "/kaggle/working/patch_index.csv"
RESULTS_DIR    = "/kaggle/working/clustering"
K_RANGE        = [4, 6, 8, 10, 12]  # K-means: try these K values
HDBSCAN_MIN_SAMPLES   = 10    # lower = more clusters/noise sensitivity
HDBSCAN_MIN_CLUST_SIZ = 50    # minimum patches to form a cluster
RANDOM_STATE   = 42
# ────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

def scale_embeddings(embeddings):
    """Standardize before clustering — important for HDBSCAN."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    print(f" Embeddings standardized: mean≈0, std≈1")
    return scaled, scaler

# ── K-MEANS ──────────────────────────────────────────────────
def run_kmeans_sweep(embeddings_scaled):
    """
    Try multiple K values, pick best by silhouette score.
    Uses MiniBatchKMeans for speed with larger datasets.
    """
    print("\n── K-Means Sweep ──")
    results = {}
    best_k, best_score, best_labels = None, -1, None

    for k in K_RANGE:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             batch_size=2048, n_init=10)
        labels = km.fit_predict(embeddings_scaled)

        # Silhouette on a subsample (full computation too slow for large N)
        n_sample = min(5000, len(embeddings_scaled))
        idx      = np.random.choice(len(embeddings_scaled), n_sample, replace=False)
        sil      = silhouette_score(embeddings_scaled[idx], labels[idx],
                                    sample_size=n_sample, random_state=RANDOM_STATE)
        db       = davies_bouldin_score(embeddings_scaled[idx], labels[idx])

        results[k] = {"silhouette": sil, "davies_bouldin": db,
                      "labels": labels, "model": km}
        print(f"  K={k:2d} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

        if sil > best_score:
            best_score  = sil
            best_k      = k
            best_labels = labels

    print(f"\n Best K={best_k} (silhouette={best_score:.4f})")
    return best_k, best_labels, results

# ── HDBSCAN ──────────────────────────────────────────────────
def run_hdbscan(embeddings_scaled):
    """
    HDBSCAN discovers clusters + marks rare patches as noise (-1).
    These noise points are your 'anomaly' candidates.
    """
    print("\n── HDBSCAN ──")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUST_SIZ,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",   # "eom" = excess of mass, more stable
        prediction_data=True
    )
    labels = clusterer.fit_predict(embeddings_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    noise_pct  = 100 * n_noise / len(labels)

    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise} ({noise_pct:.1f}%) ← anomaly candidates")

    for c in sorted(set(labels)):
        label = "NOISE/ANOMALY" if c == -1 else f"Cluster {c}"
        count = np.sum(labels == c)
        print(f"    {label}: {count} patches ({100*count/len(labels):.1f}%)")

    return labels, clusterer

# ── ELBOW PLOT ────────────────────────────────────────────────
def plot_kmeans_scores(results):
    ks   = sorted(results.keys())
    sils = [results[k]["silhouette"] for k in ks]
    dbs  = [results[k]["davies_bouldin"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, sils, "o-", color="steelblue", linewidth=2)
    ax1.set_xlabel("K"); ax1.set_ylabel("Silhouette Score ↑")
    ax1.set_title("K-Means: Silhouette (higher = better)")
    ax1.grid(alpha=0.3)

    ax2.plot(ks, dbs, "o-", color="tomato", linewidth=2)
    ax2.set_xlabel("K"); ax2.set_ylabel("Davies-Bouldin ↓")
    ax2.set_title("K-Means: Davies-Bouldin (lower = better)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/kmeans_scores.png", dpi=150)
    plt.show()
    print(f"  Saved: {RESULTS_DIR}/kmeans_scores.png")

# ── SAVE RESULTS ─────────────────────────────────────────────
def save_cluster_assignments(df, kmeans_labels, hdbscan_labels):
    df = df.copy()
    df["cluster_kmeans"]  = kmeans_labels
    df["cluster_hdbscan"] = hdbscan_labels
    df["is_anomaly"]      = (hdbscan_labels == -1).astype(int)

    out_path = f"{RESULTS_DIR}/cluster_assignments.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Cluster assignments saved → {out_path}")
    return df

# ── CLUSTER STABILITY: leave-one-patient-out preview ─────────
def cluster_stability_check(embeddings_scaled, df, best_k):
    """
    Quick leave-one-patient-out check:
    Re-cluster without each patient and measure label overlap.
    """
    print("\n── Leave-One-Patient-Out Stability Check ──")
    slides = df["slide_id"].unique()

    if len(slides) < 2:
        print("  Only 1 slide found — skipping LOO check.")
        return

    base_km = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    base_labels = base_km.fit_predict(embeddings_scaled)

    for hold_out in slides:
        mask_out = df["slide_id"] != hold_out
        emb_sub  = embeddings_scaled[mask_out]
        km_sub   = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        km_sub.fit(emb_sub)

        # Project held-out slide into reduced model
        mask_in   = ~mask_out
        pred_in   = km_sub.predict(embeddings_scaled[mask_in])
        true_in   = base_labels[mask_in]

        # Cluster overlap (rough alignment via mode matching)
        from scipy.stats import mode
        aligned = 0
        for c in range(best_k):
            c_mask = pred_in == c
            if c_mask.sum() > 0:
                m = mode(true_in[c_mask], keepdims=True).mode[0]
                aligned += np.sum(true_in[c_mask] == m)
        overlap = aligned / len(pred_in) if len(pred_in) > 0 else 0
        print(f"  Hold-out [{hold_out}]: cluster overlap ≈ {overlap:.2%}")

    print("  (>70% overlap = stable clusters independent of individual slides)")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load data
    embeddings = np.load(PCA_EMBED_PATH)
    df         = pd.read_csv(PATCH_INDEX)
    print(f"Loaded {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
    print(f"Slides in dataset: {df['slide_id'].unique().tolist()}")

    # Scale
    embeddings_scaled, scaler = scale_embeddings(embeddings)
    with open(f"{RESULTS_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # K-Means sweep
    best_k, kmeans_labels, km_results = run_kmeans_sweep(embeddings_scaled)
    plot_kmeans_scores(km_results)

    # HDBSCAN
    hdbscan_labels, hdbscan_model = run_hdbscan(embeddings_scaled)

    # Save all results
    df_results = save_cluster_assignments(df, kmeans_labels, hdbscan_labels)

    # Stability check
    cluster_stability_check(embeddings_scaled, df, best_k)

    # Save models
    with open(f"{RESULTS_DIR}/kmeans_model.pkl", "wb") as f:
        pickle.dump(km_results[best_k]["model"], f)
    with open(f"{RESULTS_DIR}/hdbscan_model.pkl", "wb") as f:
        pickle.dump(hdbscan_model, f)

    print(f"\n══ Phase 3 Complete ══")
    print(f"Best K (K-Means)  : {best_k}")
    print(f"HDBSCAN clusters  : {len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)}")
    print(f"Anomaly patches   : {np.sum(hdbscan_labels == -1)}")
    print(f"Results saved to  : {RESULTS_DIR}/")
    print(f"Next step         : Run phase4_spatial_maps.py")
