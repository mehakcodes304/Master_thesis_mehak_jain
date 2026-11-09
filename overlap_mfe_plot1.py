#!/usr/bin/env python3
"""
overlap_mfe_plot.py
───────────────────
Scatter plot: ground-truth stem overlap (%) vs hybridisation energy (XE).

How chimeric BAM records work
─────────────────────────────
Each chimeric read produces TWO BAM lines with the same QNAME.
There are two formats in the non-sorted filtered BAM:

  Pattern A — arm1_to_arm2 (QNAME ends in _arm1_to_arm2):
    Both FLAG=0; arm1 identified by XX=0, arm2 by XX>0.
    XD left  = arm1 side  (POS of XX=0 record)
    XD right = arm2 side  (POS of XX>0 record)

  Pattern B — arm2_to_arm1 (QNAME ends in _arm2_to_arm1):
    FLAG=2048 (supplementary) + FLAG=0 (primary); BOTH have XX=0.
    XD left  = FLAG=2048 side  (lower-POS / 5' region)
    XD right = FLAG=0 side     (higher-POS / 3' region)

XX and XY are positions within the parent chimeric read sequence, NOT genomic.
The POS field (4th SAM column) gives the true 1-based reference coordinate.

Overlap computation
───────────────────
For each chimeric read (QNAME pair):
  left_start  = POS of the XD-left record  (1-based reference)
  right_start = POS of the XD-right record (1-based reference)
  XD string: `left_bracket & right_bracket`
    left  → characters corresponding to left_start arm
    right → characters corresponding to right_start arm
  Pairing: '(' at left[k] pairs with the last unpaired '(' when ')' is seen at right[k].
  Reference position of left[k]  = left_start  + k
  Reference position of right[k] = right_start + k
  GT overlap = fraction of arm positions that fall in any GT base-pair position.

No TP/FP/XMAP labelling — just XE vs overlap% scatter with regression.

Output
──────
  <out_dir>/<rna>_overlap_mfe.pdf  — scatter + regression
  <out_dir>/<rna>_overlap_mfe.tsv  — per-read table
  <out_dir>/all_rnas_overlap_mfe.pdf  — combined (when >1 RNA)
  <out_dir>/all_rnas_overlap_mfe.tsv
  <out_dir>/<label>_grid.pdf          — grid of all RNA panels (label derived from CSV name)

Usage
─────
  python overlap_mfe_plot.py \\
      --bam  /path/to/condition1_splits.bam \\
      --csv  /path/to/stems2_ecoli_tRNA.csv \\
      --out  /path/to/output_dir \\
      [--rna tRNA_SeC_1_1]

python overlap_mfe_plot1.py --bam /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA/condition1_splits_filtered.bam --csv /mnt/ceph/mjain/software/envs/bio/stems/ecoli_rRNA/stems2_ecoli_rRNA.csv --out /mnt/ceph/mjain/software/envs/bio/dotplot/ecoli_rRNA_overlap
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pysam
except ImportError:
    raise ImportError("pysam is required: conda install -c bioconda pysam")


# ── GT pair parsing ───────────────────────────────────────────────────────────

def parse_stem_pairs(stem_str: str) -> set[tuple[int, int]]:
    """
    Parse one stems2 sub_interaction_stem cell, e.g. "[(1-8), (84-91)]".
    Antiparallel: first position of 5' range pairs with LAST of 3' range.
    Returns a set of 1-based (i, j) pairs with i < j.
    """
    ranges = re.findall(r'\((\d+)-(\d+)\)', stem_str)
    if len(ranges) != 2:
        return set()
    i_start, i_end = int(ranges[0][0]), int(ranges[0][1])
    j_start, j_end = int(ranges[1][0]), int(ranges[1][1])
    i_pos = list(range(i_start, i_end + 1))
    j_pos = list(range(j_start, j_end + 1))[::-1]  # antiparallel
    length = min(len(i_pos), len(j_pos))
    pairs = set()
    for k in range(length):
        a, b = i_pos[k], j_pos[k]
        pairs.add((min(a, b), max(a, b)))
    return pairs


def gt_pairs_for_rna(stems_df: pd.DataFrame, rna_name: str) -> set[tuple[int, int]]:
    """Return all GT base pairs for one RNA (exact match, then prefix match)."""
    sub = stems_df[stems_df["name"] == rna_name]
    if sub.empty:
        short = rna_name.split("_URS")[0]
        sub = stems_df[stems_df["name"].str.startswith(short)]
    if sub.empty:
        return set()
    pairs: set[tuple[int, int]] = set()
    for stem_str in sub["sub_interaction_stem"]:
        pairs |= parse_stem_pairs(str(stem_str))
    return pairs


# ── XD pair extraction ────────────────────────────────────────────────────────

def parse_xd_pairs(xd: str, arm1_start: int, arm2_start: int) -> list[tuple[int, int]]:
    """
    Stack-based XD parser.
    arm1_start / arm2_start: 1-based reference starts for each arm.
    Returns list of 1-based (i, j) pairs with i < j.
    """
    if "&" not in xd:
        return []
    left, right = xd.split("&", 1)
    arm1_pos = [arm1_start + k for k in range(len(left))]
    arm2_pos = [arm2_start + k for k in range(len(right))]
    stack, pairs = [], []
    for k, ch in enumerate(left):
        if ch == "(":
            stack.append(arm1_pos[k])
    for k, ch in enumerate(right):
        if ch == ")" and stack:
            a1 = stack.pop()
            a2 = arm2_pos[k]
            pairs.append((min(a1, a2), max(a1, a2)))
    return pairs


# ── Overlap computation ───────────────────────────────────────────────────────

def arm_overlap_frac(arm_start: int, arm_len: int, gt_positions: set[int]) -> float:
    """Fraction of arm nucleotides falling in gt_positions."""
    if arm_len == 0:
        return 0.0
    arm_set = set(range(arm_start, arm_start + arm_len))
    return len(arm_set & gt_positions) / arm_len


# ── BAM reader ────────────────────────────────────────────────────────────────

def read_overlap_data(
    bam_path: str,
    rna_name: str,
    gt_pairs: set[tuple[int, int]],
) -> pd.DataFrame:
    """
    Reads the non-sorted BAM linearly (no index needed).

    Two QNAME patterns exist in the filtered BAM:

      Pattern A — arm1_to_arm2  (both FLAG=0):
        XX=0 record  → XD left  arm (left_start  = POS)
        XX>0 record  → XD right arm (right_start = POS)

      Pattern B — arm2_to_arm1  (FLAG=2048 + FLAG=0, both XX=0):
        FLAG=2048 record → XD left  arm (left_start  = POS)
        FLAG=0    record → XD right arm (right_start = POS)

    Records whose reference name does not match rna_name are ignored;
    reads where only one arm maps here are skipped (cross-RNA chimeras).

    Returns a DataFrame with one row per chimeric read.
    """
    gt_paired_pos = {i for i, j in gt_pairs} | {j for i, j in gt_pairs}

    # Find the reference name as it appears in the BAM header
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    bam_refs = set(bam.references)
    if rna_name in bam_refs:
        target_ref = rna_name
    else:
        short = rna_name.split("_URS")[0]
        candidates = [r for r in bam_refs if short in r]
        if not candidates:
            bam.close()
            return pd.DataFrame()
        target_ref = candidates[0]

    # Bucket by QNAME into {"left": rec, "right": rec}
    # "left"  = the record whose POS maps to XD left  side (before &)
    # "right" = the record whose POS maps to XD right side (after  &)
    buckets: dict[str, dict] = defaultdict(lambda: {"left": None, "right": None})

    for read in bam.fetch(until_eof=True):
        if read.is_unmapped:
            continue
        if read.reference_name != target_ref:
            continue
        if not (read.has_tag("XX") and read.has_tag("XD") and read.has_tag("XE")):
            continue

        xx  = int(read.get_tag("XX"))
        qn  = read.query_name
        flg = read.flag

        if "arm1_to_arm2" in qn:
            # Pattern A: both FLAG=0; XX=0 → left, XX>0 → right
            if xx == 0:
                if buckets[qn]["left"] is None:
                    buckets[qn]["left"] = read
            else:
                existing = buckets[qn]["right"]
                if existing is None or xx > int(existing.get_tag("XX")):
                    buckets[qn]["right"] = read
        elif "arm2_to_arm1" in qn:
            # Pattern B: FLAG=2048 → left, FLAG=0 → right
            if flg == 2048:
                if buckets[qn]["left"] is None:
                    buckets[qn]["left"] = read
            elif flg == 0:
                if buckets[qn]["right"] is None:
                    buckets[qn]["right"] = read
        else:
            # Unknown pattern: fall back to XX-based heuristic
            if xx == 0:
                if buckets[qn]["left"] is None:
                    buckets[qn]["left"] = read
            else:
                if buckets[qn]["right"] is None:
                    buckets[qn]["right"] = read

    bam.close()

    # Build one row per chimeric read that has BOTH arms on this RNA
    rows = []
    for qn, rec in buckets.items():
        left_rec  = rec["left"]
        right_rec = rec["right"]

        if left_rec is None or right_rec is None:
            # One arm maps elsewhere — skip for this RNA's plot
            continue

        left_start  = left_rec.reference_start  + 1   # 0-based pysam → 1-based
        right_start = right_rec.reference_start + 1

        try:
            xe = float(left_rec.get_tag("XE"))
            xd = left_rec.get_tag("XD")
        except KeyError:
            continue

        if "&" not in xd:
            continue

        left_seq, right_seq = xd.split("&", 1)
        left_len  = len(left_seq)
        right_len = len(right_seq)

        # Overlap fractions against merged set of all GT-paired positions
        l_frac = arm_overlap_frac(left_start,  left_len,  gt_paired_pos)
        r_frac = arm_overlap_frac(right_start, right_len, gt_paired_pos)
        agg_pct = (l_frac + r_frac) / 2 * 100

        # XD-derived pairs (for optional inspection column)
        pairs      = parse_xd_pairs(xd, left_start, right_start)
        n_gt_exact = sum(1 for p in pairs if p in gt_pairs)

        rows.append({
            "qname":                 qn,
            "xe":                    xe,
            "left_start":            left_start,
            "right_start":           right_start,
            "left_len":              left_len,
            "right_len":             right_len,
            "left_overlap_pct":      round(l_frac  * 100, 2),
            "right_overlap_pct":     round(r_frac  * 100, 2),
            "aggregate_overlap_pct": round(agg_pct, 2),
            "n_xd_pairs":            len(pairs),
            "n_xd_gt_exact":         n_gt_exact,
        })

    return pd.DataFrame(rows)


# ── Plot ──────────────────────────────────────────────────────────────────────

def draw_overlap_mfe(rna_name: str, df: pd.DataFrame, out_path: Path) -> None:
    """
    Simple scatter plot: aggregate GT overlap (%) vs hybridisation energy (XE).
    Includes linear regression line and Pearson r annotation.
    """
    short = rna_name.split("_URS")[0]
    x = df["aggregate_overlap_pct"].values
    y = df["xe"].values

    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter
    ax.scatter(x, y, alpha=0.45, s=18, color="#2c7bb6", edgecolors="none",
               label=f"n = {len(df)}")

    # Linear regression
    if len(df) >= 3:
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, m * xline + b, color="#d7191c", linewidth=1.4,
                label=f"y = {m:.3f}x + {b:.2f}")
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.03, 0.82,
                f"Pearson r = {r:.3f}",
                transform=ax.transAxes, fontsize=9,
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Aggregate GT stem overlap  (%)  ←  (left_overlap + right_overlap) / 2",
                  fontsize=9)
    ax.set_ylabel("Hybridisation energy XE  (kcal/mol)", fontsize=9)
    ax.set_xlim(-2, 102)
    for xv in [25, 50, 75]:
        ax.axvline(xv, color="lightgrey", linewidth=0.6, linestyle=":")

    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.set_title(f"{short} — GT overlap (%) vs hybridisation energy (MFE)",
                 fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {out_path.name}")


# ── Grid figure ───────────────────────────────────────────────────────────────

def draw_grid(rna_data: list[tuple[str, pd.DataFrame]], out_path: Path,
              title: str = "") -> None:
    """
    All-RNAs grid: one subplot per RNA, tiled into a single figure.
    rna_data: list of (rna_name, df) pairs (skip empty dfs beforehand).
    """
    n     = len(rna_data)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             squeeze=False)

    for idx, (rna_name, df) in enumerate(rna_data):
        row, col = divmod(idx, ncols)
        ax   = axes[row][col]
        short = rna_name.split("_URS")[0]
        x    = df["aggregate_overlap_pct"].values
        y    = df["xe"].values

        ax.scatter(x, y, alpha=0.4, s=8, color="#2c7bb6", edgecolors="none")

        if len(df) >= 3:
            m, b   = np.polyfit(x, y, 1)
            xline  = np.linspace(x.min(), x.max(), 200)
            ax.plot(xline, m * xline + b, color="#d7191c", linewidth=1.0)
            r      = np.corrcoef(x, y)[0, 1]
            ax.text(0.97, 0.97,
                    f"r={r:.2f}\nn={len(df)}",
                    transform=ax.transAxes, fontsize=7,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlim(-2, 102)
        ax.set_title(short, fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("GT overlap (%)", fontsize=7)
        ax.set_ylabel("XE (kcal/mol)", fontsize=7)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Grid → {out_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bam", required=True,
                   help="Non-sorted filtered chimeric BAM (no index needed)")
    p.add_argument("--csv", required=True,
                   help="stems2 CSV with sub_interaction_stem column")
    p.add_argument("--out", required=True,
                   help="Output directory")
    p.add_argument("--rna", default=None,
                   help="Single RNA name (e.g. tRNA_SeC_1_1); default = all in CSV")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Overlap-MFE scatter analysis")
    print(f"{'='*64}")
    print(f"  BAM : {args.bam}")
    print(f"  CSV : {args.csv}")

    stems_df = pd.read_csv(args.csv)

    if args.rna:
        short_filter = args.rna.split("_URS")[0]
        rna_names = [n for n in stems_df["name"].unique()
                     if n == args.rna or n.startswith(short_filter)]
        if not rna_names:
            print(f"ERROR: '{args.rna}' not found in CSV")
            return
    else:
        rna_names = list(stems_df["name"].unique())

    all_rows = []

    for rna_name in rna_names:
        short = rna_name.split("_URS")[0]
        row0  = stems_df.loc[stems_df["name"] == rna_name]
        seq_len = int(row0["seq_len"].iloc[0]) if not row0.empty else "?"

        print(f"\n[{short}]  ({seq_len} nt)")

        gt_pairs = gt_pairs_for_rna(stems_df, rna_name)
        n_stems  = stems_df[stems_df["name"] == rna_name].shape[0]
        print(f"  GT pairs : {len(gt_pairs)}  (from {n_stems} stems)")

        df = read_overlap_data(args.bam, rna_name, gt_pairs)
        if df.empty:
            print(f"  No reads with both arms on this RNA — skipping")
            continue

        print(f"  Reads (both arms on RNA) : {len(df)}")
        print(f"  Mean aggregate overlap   : {df['aggregate_overlap_pct'].mean():.1f}%")
        print(f"  Mean XE                  : {df['xe'].mean():.2f} kcal/mol")
        if len(df) >= 3:
            r = np.corrcoef(df["aggregate_overlap_pct"], df["xe"])[0, 1]
            print(f"  Pearson r (overlap~XE)   : {r:.4f}")

        tsv_path = out_dir / f"{short}_overlap_mfe.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)
        print(f"  TSV → {tsv_path.name}")

        pdf_path = out_dir / f"{short}_overlap_mfe.pdf"
        draw_overlap_mfe(rna_name, df, pdf_path)

        df["rna"] = short
        all_rows.append((rna_name, df))

    if len(all_rows) > 1:
        # Combined single-scatter (all reads merged)
        combined     = pd.concat([df for _, df in all_rows], ignore_index=True)
        combined_tsv = out_dir / "all_rnas_overlap_mfe.tsv"
        combined.to_csv(combined_tsv, sep="\t", index=False)
        combined_pdf = out_dir / "all_rnas_overlap_mfe.pdf"
        draw_overlap_mfe("All RNAs (combined)", combined, combined_pdf)
        r = np.corrcoef(combined["aggregate_overlap_pct"], combined["xe"])[0, 1]
        print(f"\n  Combined Pearson r : {r:.4f}")
        print(f"  Combined TSV → {combined_tsv.name}")
        print(f"  Combined PDF → {combined_pdf.name}")

        # Grid figure — each RNA as its own panel
        # Name the grid after the CSV so each dataset gets a distinct file,
        # e.g. stems2_ecoli_tRNA.csv  → ecoli_tRNA_grid.pdf
        csv_stem = Path(args.csv).stem          # e.g. stems2_ecoli_tRNA
        label = csv_stem.replace("stems2_", "")  # e.g. ecoli_tRNA
        grid_pdf = out_dir / f"{label}_grid.pdf"
        draw_grid(all_rows, grid_pdf, title=label.replace("_", " "))
        print(f"  Grid PDF  → {grid_pdf.name}")

    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()