#!/usr/bin/env python3
"""
plot_contact_heatmap.py
───────────────────────
Plots intramolecular contact frequency heatmaps from the CSV/TSV outputs
of build_contact_matrix.py.

For each RNA:
  1. Full N×N heatmap of the entire contact matrix
  2. Zoomed heatmap per detected stem region (from the sparse TSV)

Colour scale uses log₁₊ₓ(count) because raw counts span 1–240 and a
linear scale makes low-frequency contacts invisible.
White = no contact observed (count = 0).

Usage
─────
  conda activate bio
  pip install matplotlib seaborn   # if not already installed
  python visualisation.py \\
      --matrix_dir ./contact_matrices/ecoli_rRNA \\
      --out        ./contact_matrices/ecoli_rRNA/plots \\
      [--min_support 10]   # threshold for zoomed plots
      [--group_gap 3]      # gap (nt) for grouping arm positions

python visualisation.py   --matrix_dir ./contact_matrices/ecoli_rRNA   --out ./contact_matrices/ecoli_rRNA/plots   --min_support 10   --group_gap 3

Dependencies: matplotlib, numpy, pandas
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe for server/VM use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--matrix_dir",  required=True,
                   help="Directory with *_contact_matrix.csv and "
                        "*_contact_matrix_sparse.tsv files")
    p.add_argument("--out",         required=True,
                   help="Output directory for PNG plots (created if absent)")
    p.add_argument("--min_support", type=int, default=10,
                   help="Min read count to show a region in zoomed plot "
                        "(default: 10)")
    p.add_argument("--group_gap",   type=int, default=3,
                   help="Max gap (nt) to merge positions into one arm region "
                        "(default: 3)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def rna_short_name(filename_stem: str) -> str:
    """Strip URS identifier from filename stem for plot titles."""
    return filename_stem.replace("_contact_matrix", "").split("_URS")[0]


def group_consecutive(positions: list, gap: int = 3) -> list:
    """
    Group sorted positions into contiguous regions allowing gaps up to `gap` nt.
    Returns list of (start, end) tuples — 1-based inclusive.
    """
    if not positions:
        return []
    groups = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [(g[0], g[-1]) for g in groups]


def get_arm_pairs(sparse_df: pd.DataFrame, min_support: int,
                  group_gap: int) -> list:
    """
    From the sparse TSV, find interacting arm region pairs above min_support.
    Returns list of dicts with arm1_start/end, arm2_start/end, total_support.
    """
    hi = sparse_df[sparse_df["count"] >= min_support].copy()
    if hi.empty:
        return []

    a1_regs = group_consecutive(sorted(hi["pos_arm1"].unique()), group_gap)
    a2_regs = group_consecutive(sorted(hi["pos_arm2"].unique()), group_gap)

    pairs = []
    for a1s, a1e in a1_regs:
        for a2s, a2e in a2_regs:
            sub = hi[
                hi["pos_arm1"].between(a1s, a1e) &
                hi["pos_arm2"].between(a2s, a2e)
            ]
            if sub.empty:
                continue
            pairs.append(dict(
                arm1_start=a1s, arm1_end=a1e,
                arm2_start=a2s, arm2_end=a2e,
                total_support=int(sub["count"].sum()),
            ))
    return sorted(pairs, key=lambda x: -x["total_support"])


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: FULL HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_full(matrix: np.ndarray, short_name: str, out_path: Path) -> None:
    """
    Full N×N contact matrix heatmap.
    log₁₊ₓ colour scale; white = zero.
    """
    n       = matrix.shape[0]
    log_mat = np.log1p(matrix.astype(float))

    # Tick spacing
    tick_step = 10 if n <= 200 else 50 if n <= 500 else 100
    ticks     = list(range(0, n, tick_step))
    labels    = [str(t + 1) for t in ticks]

    fig_size = max(6, min(16, n / 60))
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "contact",
        ["white", "#AED6F1", "#2874A6", "#1A5276", "#922B21", "#E74C3C"],
    )
    im = ax.imshow(log_mat, cmap=cmap, origin="lower",
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Position on reference — arm2 (x-axis, 1-based)", fontsize=10)
    ax.set_ylabel("Position on reference — arm1 (y-axis, 1-based)", fontsize=10)
    ax.set_title(
        f"{short_name}  —  Intramolecular contact frequency\n"
        f"colour = log₁₊ₓ(read count)  |  white = no contact",
        fontsize=10, pad=8,
    )

    # Colourbar with raw count reference ticks
    cbar      = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    max_count = int(matrix.max())
    raw_ticks = [v for v in [0, 1, 5, 10, 25, 50, 100, max_count]
                 if v <= max_count]
    cbar.set_ticks([np.log1p(v) for v in raw_ticks])
    cbar.set_ticklabels([str(v) for v in raw_ticks], fontsize=7)
    cbar.set_label("read count (raw)", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: ZOOMED REGION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_zoom(
    matrix:     np.ndarray,
    short_name: str,
    arm1_start: int, arm1_end: int,
    arm2_start: int, arm2_end: int,
    label:      str,
    out_path:   Path,
) -> None:
    """
    Zoomed heatmap around one detected stem interaction.
    Pads 10 nt either side of each arm for structural context.
    """
    n   = matrix.shape[0]
    pad = 10
    r1s = max(0, arm1_start - 1 - pad)
    r1e = min(n, arm1_end + pad)
    r2s = max(0, arm2_start - 1 - pad)
    r2e = min(n, arm2_end + pad)

    sub     = np.log1p(matrix[r1s:r1e, r2s:r2e].astype(float))
    rlabels = [str(i + 1) for i in range(r1s, r1e)]
    clabels = [str(i + 1) for i in range(r2s, r2e)]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap    = mcolors.LinearSegmentedColormap.from_list(
        "contact", ["white", "#AED6F1", "#2874A6", "#E74C3C"]
    )
    im = ax.imshow(sub, cmap=cmap, origin="lower", aspect="auto")

    # Tick every 5 positions for readability
    step = max(1, (len(clabels) + len(rlabels)) // 20)
    xt   = list(range(0, len(clabels), step))
    yt   = list(range(0, len(rlabels), step))
    ax.set_xticks(xt)
    ax.set_xticklabels([clabels[i] for i in xt], rotation=90, fontsize=8)
    ax.set_yticks(yt)
    ax.set_yticklabels([rlabels[i] for i in yt], fontsize=8)
    ax.set_xlabel("Arm 2 position (1-based)", fontsize=10)
    ax.set_ylabel("Arm 1 position (1-based)", fontsize=10)
    ax.set_title(
        f"{short_name}  —  {label}\n"
        f"arm1: {arm1_start}–{arm1_end}  ↔  arm2: {arm2_start}–{arm2_end}",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.03).set_label(
        "log₁₊ₓ(count)", fontsize=8
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_dir = Path(args.matrix_dir)
    csv_files  = sorted(matrix_dir.glob("*_contact_matrix.csv"))

    if not csv_files:
        print(f"ERROR: no *_contact_matrix.csv files found in {matrix_dir}")
        return

    print(f"\n{'='*55}")
    print(f"  Contact Matrix Heatmaps")
    print(f"{'='*55}")
    print(f"  Input dir:   {matrix_dir}")
    print(f"  Output dir:  {out_dir}")
    print(f"  Min support: {args.min_support}")

    for csv_path in csv_files:
        stem       = csv_path.stem                        # e.g. rRNA_5S_URS..._contact_matrix
        short      = rna_short_name(stem)
        sparse_path = matrix_dir / (
            stem.replace("_contact_matrix", "_contact_matrix_sparse") + ".tsv"
        )

        print(f"\n  [{short}]")

        # Load matrix
        df_mat = pd.read_csv(csv_path, index_col=0)
        matrix = df_mat.values.astype(np.int32)
        n      = matrix.shape[0]
        print(f"    {n}×{n} matrix  |  max count: {matrix.max()}"
              f"  |  non-zero cells: {np.count_nonzero(matrix):,}")

        # Full heatmap
        full_path = out_dir / f"{short}_heatmap_full.png"
        plot_full(matrix, short, full_path)
        print(f"    Full heatmap  → {full_path.name}")

        # Zoomed heatmaps from sparse TSV
        if not sparse_path.exists():
            print(f"    Sparse TSV not found — skipping zoomed plots")
            continue

        sparse_df = pd.read_csv(sparse_path, sep="\t")
        arm_pairs = get_arm_pairs(sparse_df, args.min_support, args.group_gap)
        print(f"    Arm-pair regions detected (≥{args.min_support} reads): "
              f"{len(arm_pairs)}")

        for i, ap in enumerate(arm_pairs[:8], 1):    # max 8 zoomed plots per RNA
            label     = f"interaction_{i}"
            zoom_path = out_dir / f"{short}_zoom_{label}.png"
            plot_zoom(
                matrix, short,
                ap["arm1_start"], ap["arm1_end"],
                ap["arm2_start"], ap["arm2_end"],
                label, zoom_path,
            )
            print(f"    Zoomed [{label}]  "
                  f"arm1 {ap['arm1_start']}–{ap['arm1_end']}  ↔  "
                  f"arm2 {ap['arm2_start']}–{ap['arm2_end']}  "
                  f"(support={ap['total_support']})  → {zoom_path.name}")

    print(f"\n{'='*55}")
    print(f"  Done. Plots saved to: {out_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()