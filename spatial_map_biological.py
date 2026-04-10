# ============================================================
# PHASE 4: Spatial Maps + Biological Validation
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────
CLUSTER_CSV  = "/kaggle/working/clustering/cluster_assignments.csv"
RESULTS_DIR  = "/kaggle/working/spatial_maps"
PATCH_SIZE   = 224        # must match Phase 1
# Optional: path to metadata CSV with columns [slide_id, age, sex, group]
METADATA_CSV = None       # set to e.g. "/kaggle/input/your-dataset/metadata.csv"
# ────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── SPATIAL MAP ──────────────────────────────────────────────
def plot_spatial_map(df_slide, slide_id, cluster_col="cluster_kmeans",
                     cmap_name="tab10"):
    """
    Project cluster labels back onto slide coordinates.
    Each colored square = one patch, colored by cluster ID.
    """
    labels  = df_slide[cluster_col].values
    xs      = df_slide["x"].values
    ys      = df_slide["y"].values

    unique_labels = sorted(set(labels))
    cmap   = plt.get_cmap(cmap_name, len(unique_labels))
    colors = {l: cmap(i) for i, l in enumerate(unique_labels)}

    # Grid size
    x_max  = xs.max() + PATCH_SIZE
    y_max  = ys.max() + PATCH_SIZE
    fig_w  = max(8, x_max / 100)
    fig_h  = max(6, y_max / 100)

    fig, ax = plt.subplots(figsize=(min(fig_w, 20), min(fig_h, 16)))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for x, y, label in zip(xs, ys, labels):
        color = colors[label]
        rect  = mpatches.Rectangle(
            (x, y_max - y - PATCH_SIZE),    # flip Y for natural orientation
            PATCH_SIZE, PATCH_SIZE,
            linewidth=0, facecolor=color, alpha=0.85
        )
        ax.add_patch(rect)

    # Legend
    legend_patches = []
    for l in unique_labels:
        name = "Anomaly/Noise" if l == -1 else f"State {l}"
        legend_patches.append(
            mpatches.Patch(color=colors[l], label=name)
        )
    ax.legend(handles=legend_patches, loc="upper right",
              framealpha=0.3, labelcolor="white",
              facecolor="#2d2d44", fontsize=8)

    ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
    ax.set_xlabel("X (patches)", color="white")
    ax.set_ylabel("Y (patches)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    title = f"Spatial Map — {slide_id}\n({cluster_col})"
    ax.set_title(title, color="white", fontsize=11, pad=10)

    out = f"{RESULTS_DIR}/spatial_{slide_id}_{cluster_col}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Saved: {out}")

# ── CLUSTER COMPOSITION PER SLIDE ────────────────────────────
def plot_cluster_composition(df, cluster_col="cluster_kmeans"):
    """
    Stacked bar: what fraction of each slide belongs to each cluster?
    Helps spot if one slide is dominated by an unusual state.
    """
    slides       = df["slide_id"].unique()
    cluster_ids  = sorted(df[cluster_col].unique())
    cmap         = plt.get_cmap("tab10", len(cluster_ids))

    proportions = {}
    for s in slides:
        sub    = df[df["slide_id"] == s]
        counts = sub[cluster_col].value_counts()
        proportions[s] = {c: counts.get(c, 0) / len(sub) for c in cluster_ids}

    fig, ax = plt.subplots(figsize=(max(6, len(slides) * 1.5), 5))
    bottoms = np.zeros(len(slides))
    slide_list = list(slides)

    for i, c in enumerate(cluster_ids):
        vals = [proportions[s][c] for s in slide_list]
        label = "Anomaly" if c == -1 else f"State {c}"
        ax.bar(slide_list, vals, bottom=bottoms, color=cmap(i),
               label=label, edgecolor="white", linewidth=0.3)
        bottoms += np.array(vals)

    ax.set_ylabel("Fraction of patches")
    ax.set_title(f"Cluster Composition per Slide ({cluster_col})")
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/composition_{cluster_col}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Saved: {out}")
    return pd.DataFrame(proportions).T

# ── ANOMALY SUMMARY ──────────────────────────────────────────
def anomaly_summary(df):
    """
    Report anomaly (HDBSCAN noise) patches per slide.
    This is the 'distinctive patient' detection step.
    """
    print("\n── Anomaly Detection Summary (HDBSCAN noise patches) ──")
    summary = df.groupby("slide_id")["is_anomaly"].agg(
        total_patches="count",
        anomaly_patches="sum"
    )
    summary["anomaly_pct"] = 100 * summary["anomaly_patches"] / summary["total_patches"]
    summary = summary.sort_values("anomaly_pct", ascending=False)
    print(summary.to_string())

    # Flag slides with unusually high anomaly %
    mean_a = summary["anomaly_pct"].mean()
    std_a  = summary["anomaly_pct"].std()
    flagged = summary[summary["anomaly_pct"] > mean_a + std_a]
    if len(flagged) > 0:
        print(f"\n  Slides with anomaly% > mean+1σ ({mean_a:.1f}% + {std_a:.1f}%):")
        for sid, row in flagged.iterrows():
            print(f"    → {sid}: {row['anomaly_pct']:.1f}% anomalous patches "
                  f"← possible distinctive biological state")
    else:
        print("\n  No strongly anomalous slides detected.")
    return summary

# ── METADATA CORRELATION ─────────────────────────────────────
def metadata_correlation(composition_df, metadata_csv):
    """
    Correlate cluster proportions with age/sex if metadata available.
    This is a POST-HOC analysis — metadata was never used during clustering.
    """
    if metadata_csv is None or not Path(metadata_csv).exists():
        print("\n  No metadata CSV provided — skipping metadata correlation.")
        print("    Set METADATA_CSV path if you have age/sex/group data.")
        return

    meta = pd.read_csv(metadata_csv)
    merged = composition_df.merge(meta, left_index=True, right_on="slide_id", how="inner")

    print("\n── Metadata Correlation ──")
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    cluster_cols = [c for c in numeric_cols if str(c).startswith(("0","1","2","3","4","5","6","7","8","9","-"))]
    meta_numeric = [c for c in ["age"] if c in merged.columns]

    if not meta_numeric:
        print("  No numeric metadata columns (age) found.")
        return

    from scipy.stats import pearsonr, spearmanr
    print(f"  {'Cluster':<12} {'Meta':<8} {'Pearson r':>10} {'p-value':>10}")
    print("  " + "-" * 44)
    for cc in cluster_cols:
        for mc in meta_numeric:
            valid = merged[[cc, mc]].dropna()
            if len(valid) < 4:
                continue
            r, p = spearmanr(valid[cc], valid[mc])
            flag = " ← !" if p < 0.05 else ""
            print(f"  State {str(cc):<6} {mc:<8} {r:>10.3f} {p:>10.4f}{flag}")

# ── NUISANCE VARIABLE CHECK ───────────────────────────────────
def nuisance_check(df, cluster_col="cluster_kmeans"):
    """
    Check if clusters simply separate patients (bad — means
    the model learned patient identity, not biology).
    Good result: clusters are distributed across multiple slides.
    """
    print(f"\n── Nuisance Check: Cluster vs Slide ID ({cluster_col}) ──")
    ct = pd.crosstab(df[cluster_col], df["slide_id"], normalize="index") * 100
    print(ct.round(1).to_string())
    print("\n  Interpretation: if any cluster is >90% from one slide,")
    print("  that cluster may reflect patient/scanner bias, not biology.")

    fig, ax = plt.subplots(figsize=(max(8, len(df["slide_id"].unique()) + 2), 4))
    ct.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("% patches from slide")
    ax.set_title("Cluster Source Distribution (Nuisance Check)")
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.2, 1))
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/nuisance_check_{cluster_col}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Saved: {out}")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    df = pd.read_csv(CLUSTER_CSV)
    print(f"Loaded {len(df)} patches across {df['slide_id'].nunique()} slides")
    print(f"Columns: {df.columns.tolist()}\n")

    # 1. Spatial maps for every slide (both K-means and HDBSCAN)
    print("── Generating Spatial Maps ──")
    for slide_id in df["slide_id"].unique():
        df_s = df[df["slide_id"] == slide_id]
        plot_spatial_map(df_s, slide_id, cluster_col="cluster_kmeans")
        plot_spatial_map(df_s, slide_id, cluster_col="cluster_hdbscan")

    # 2. Cluster composition per slide
    print("\n── Cluster Composition ──")
    comp_kmeans  = plot_cluster_composition(df, cluster_col="cluster_kmeans")
    comp_hdbscan = plot_cluster_composition(df, cluster_col="cluster_hdbscan")

    # 3. Anomaly summary — distinctive patient detection
    anomaly_summary(df)

    # 4. Nuisance variable check
    nuisance_check(df, cluster_col="cluster_kmeans")

    # 5. Metadata correlation (only runs if METADATA_CSV is set)
    metadata_correlation(comp_kmeans, METADATA_CSV)

    print(f"\n══ Phase 4 Complete ══")
    print(f"All spatial maps and validation plots saved to: {RESULTS_DIR}/")
    print(f"Next step: Run phase5_visualization.py for UMAP/PCA embedding plots")
