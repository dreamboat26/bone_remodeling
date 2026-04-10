# ============================================================
# PHASE 1 (TMA-OPTIMIZED v2): .mrxs → Detect 4 Cores → Patch
#
# Fixes vs previous version:
#   1. Contour-based detection replaces Hough — finds exactly
#      the large pink blobs, not 15 false positives
#   2. All tissue filtering on CPU — eliminates CUDA OOM
#   3. Stain normalization on CPU — no GPU memory spikes
#   4. EXPECTED_CORES guard — warns if wrong number found
#   5. Saves detection preview so you can verify before patching
# ============================================================
# INSTALL (run once):
#   !apt-get install -y openslide-tools libopenslide-dev -q
#   !pip install openslide-python torchstain opencv-python-headless -q

import os, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import openslide
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG — only change these ───────────────────────────────
SLIDE_DIR      = "/kaggle/input/datasets/yepp26/check1/041"  # ← from cell_fix_structure.py
OUTPUT_DIR     = "/kaggle/working/patches"
PATCH_SIZE     = 224
LEVEL          = 1        # 1 = half resolution (good balance)
THUMB_SIZE     = 2048     # thumbnail resolution for detection
TISSUE_THRESH  = 0.20     # fraction of pink/non-white pixels to keep patch
EXPECTED_CORES = 4        # ← SET THIS: how many cores per slide

# Contour size filter (in thumbnail pixels²)
# If detecting too many: increase MIN_CORE_AREA
# If missing cores:      decrease MIN_CORE_AREA
MIN_CORE_AREA  = 3000
MAX_CORE_AREA  = 600000
# ────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── STAIN NORMALIZER (CPU) ────────────────────────────────────
def build_stain_normalizer():
    """CPU-based Macenko — zero GPU memory used."""
    try:
        import torchstain, torch
        norm = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        print(" Macenko stain normalizer ready (CPU)")

        def normalize(patch_np):
            try:
                t = torch.from_numpy(patch_np).permute(2,0,1).float()
                t_n, _, _ = norm.normalize(t, stains=False)
                # clamp to valid range, convert (3,H,W) → (H,W,3) uint8
                out = t_n.permute(1,2,0).clamp(0,255).to(torch.uint8).numpy()
                if out.ndim == 3 and out.shape[2] == 3:
                    return out
                return patch_np   # fallback if shape wrong
            except Exception:
                return patch_np   # fallback if normalization fails

        return normalize
    except Exception as e:
        print(f" Stain normalization skipped ({e})")
        return lambda x: x

# ── TISSUE FILTER (CPU) ───────────────────────────────────────
def is_tissue(patch_np, threshold=TISSUE_THRESH):
    """
    Tuned for pink H&E TMA cores.
    Detects pink staining + dark nuclei — ignores white background.
    """
    f        = patch_np.astype(np.float32) / 255.0
    r, g, b  = f[:,:,0], f[:,:,1], f[:,:,2]
    pink     = (r > 0.5) & (g < 0.75) & (b > 0.3) & (r > g)
    dark     = f.mean(axis=2) < 0.7
    return (pink | dark).mean() > threshold

# ── CONTOUR-BASED CORE DETECTION ─────────────────────────────
def detect_cores_contour(slide, thumb_size=THUMB_SIZE):
    """
    1. Get small thumbnail
    2. Isolate pink/magenta stain in HSV space
    3. Find contours of large stained blobs
    4. Fit enclosing circle to each blob
    Returns: list of (cx, cy, radius) in thumbnail coords
    """
    thumb    = slide.get_thumbnail((thumb_size, thumb_size))
    thumb_np = np.array(thumb.convert("RGB"))
    hsv      = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)

    # Pink/magenta hue ranges
    mask1 = cv2.inRange(hsv, (140, 30, 80),  (180, 255, 255))
    mask2 = cv2.inRange(hsv, (0,   30, 80),  (20,  255, 255))
    # Low-saturation stained tissue (light pink areas)
    mask3 = cv2.inRange(hsv, (0,   15, 100), (180, 255, 255))
    mask  = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

    # Morphological cleanup
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cores = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_CORE_AREA < area < MAX_CORE_AREA:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cores.append((int(cx), int(cy), int(radius)))

    # Sort top-to-bottom (matches vertical layout of your 4-core slides)
    cores.sort(key=lambda c: c[1])
    return cores, thumb_np, mask

def save_detection_preview(thumb_np, mask, cores, save_path):
    """3-panel preview: thumbnail | mask | circles drawn."""
    vis    = thumb_np.copy()
    colors = [(0,255,0),(0,200,255),(255,165,0),(255,0,255)]
    for i, (cx, cy, r) in enumerate(cores):
        col = colors[i % len(colors)]
        cv2.circle(vis, (cx, cy), r, col, 3)
        cv2.circle(vis, (cx, cy), 5, (255,255,255), -1)
        cv2.putText(vis, f"Core {i}", (cx-35, cy-r-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    combined = np.hstack([
        cv2.resize(thumb_np,  (700, 900)),
        cv2.resize(mask_rgb,  (700, 900)),
        cv2.resize(vis,       (700, 900)),
    ])
    Image.fromarray(combined).save(save_path)
    print(f"  Preview → {save_path}")
    print(f"  [Left: original | Middle: stain mask | Right: detected cores]")

# ── SCALE CORE TO SLIDE LEVEL ─────────────────────────────────
def scale_core(cx_t, cy_t, r_t, slide, thumb_size, level):
    sw, sh = slide.level_dimensions[0]
    lw, lh = slide.level_dimensions[level]
    sx, sy = sw / thumb_size, sh / thumb_size
    lx, ly = lw / sw,         lh / sh
    return int(cx_t*sx*lx), int(cy_t*sy*ly), int(r_t*sx*lx)

# ── PATCH ONE CORE ────────────────────────────────────────────
def patch_core(slide, cx, cy, r, level, patch_size,
               stain_normalize, save_dir, core_id):
    """Extract patches inside circle boundary. CPU only."""
    ds  = int(slide.level_downsamples[level])
    x0  = max(0, cx - r)
    y0  = max(0, cy - r)

    positions = [
        (x, y)
        for y in range(y0, cy + r - patch_size + 1, patch_size)
        for x in range(x0, cx + r - patch_size + 1, patch_size)
        if (x + patch_size//2 - cx)**2 + (y + patch_size//2 - cy)**2 <= r**2
    ]

    saved = []
    for (x, y) in positions:
        region   = slide.read_region((x*ds, y*ds), level,
                                     (patch_size, patch_size))
        patch_np = np.array(region.convert("RGB"))
        if not is_tissue(patch_np):
            continue
        patch_np = stain_normalize(patch_np)
        fname    = save_dir / f"core{core_id}_x{x}_y{y}.png"
        Image.fromarray(patch_np).save(fname)
        saved.append((str(fname), core_id, x, y))
    return saved

# ── PROCESS ONE SLIDE ─────────────────────────────────────────
def process_slide(slide_path, stain_normalize):
    name      = Path(slide_path).stem
    save_dir  = Path(OUTPUT_DIR) / name
    done_flag = save_dir / ".done"

    if done_flag.exists():
        existing = list(save_dir.glob("*.png"))
        print(f"  ↷ {name} already done ({len(existing)} patches)")
        records = []
        for p in existing:
            if "cores_detected" in p.stem:
                continue
            parts = p.stem.split("_")
            try:
                records.append((str(p),
                    int(parts[0].replace("core","")),
                    int(parts[1][1:]), int(parts[2][1:])))
            except Exception:
                pass
        return records

    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        slide = openslide.OpenSlide(str(slide_path))
    except Exception as e:
        print(f"  Cannot open {name}: {e}")
        return []

    lw, lh = slide.level_dimensions[LEVEL]
    print(f"\n── {name}  [{lw}×{lh} @ level {LEVEL}]")

    cores, thumb_np, mask = detect_cores_contour(slide, THUMB_SIZE)
    save_detection_preview(thumb_np, mask, cores,
                           save_dir / f"{name}_cores_detected.png")

    print(f"  Detected {len(cores)} cores", end="  ")
    if len(cores) != EXPECTED_CORES:
        print(f" Expected {EXPECTED_CORES} — got {len(cores)}")
        print(f"  Adjust MIN_CORE_AREA/MAX_CORE_AREA in CONFIG and re-run")
        if len(cores) == 0:
            slide.close()
            return []
        # Keep going with however many were found
    else:
        print("✓")

    t0, records = time.time(), []
    for i, (cx_t, cy_t, r_t) in enumerate(cores):
        cx, cy, r = scale_core(cx_t, cy_t, r_t, slide, THUMB_SIZE, LEVEL)
        print(f"  Core {i}: centre=({cx},{cy})  radius={r}px", end=" → ")
        patches = patch_core(slide, cx, cy, r, LEVEL, PATCH_SIZE,
                             stain_normalize, save_dir, core_id=i)
        print(f"{len(patches)} patches")
        records.extend(patches)

    slide.close()
    done_flag.touch()
    print(f"  ✓ {len(records)} total patches | {time.time()-t0:.1f}s")
    return records


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()

    slides = list(Path(SLIDE_DIR).rglob("*.mrxs"))
    if not slides:
        raise FileNotFoundError(f"No .mrxs files in {SLIDE_DIR}")
    print(f"Found {len(slides)} slides | Expected {EXPECTED_CORES} cores each\n")

    stain_normalize = build_stain_normalizer()

    all_records = []
    for slide_path in slides:
        for (path, core_id, x, y) in process_slide(str(slide_path),
                                                     stain_normalize):
            all_records.append({
                "path":     path,
                "slide_id": Path(path).parent.name,
                "core_id":  core_id,
                "x":        x,
                "y":        y,
            })

    df = pd.DataFrame(all_records)
    df.to_csv("/kaggle/working/patch_index.csv", index=False)

    print(f"\n══ Phase 1 Complete ══")
    print(f"Slides        : {len(slides)}")
    if len(df):
        for sid, grp in df.groupby("slide_id"):
            print(f"\n  {sid}:")
            for cid, cnt in grp.groupby("core_id")["path"].count().items():
                print(f"    Core {cid}: {cnt} patches")
        print(f"\nTotal patches : {len(df)}")
    print(f"Time          : {(time.time()-t_start)/60:.1f} min")
    print(f"Index saved   : /kaggle/working/patch_index.csv")
    print(f"\n Open *_cores_detected.png in each slide folder to verify")
    print(f"  correct cores were found before running Phase 2")
    print(f"Next step     : Run phase2_feature_extraction.py")
