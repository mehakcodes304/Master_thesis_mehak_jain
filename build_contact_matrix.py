#!/usr/bin/env python3
"""
build_contact_matrix.py
───────────────────────
Builds a per-RNA intramolecular contact frequency matrix from the
filtered splits BAM produced by filter_splits_bam.py.

For each RNA (5S rRNA, 16S rRNA, 23S rRNA, tRNA-Ala, etc.) separately:
  - Creates an N×N matrix where N = length of that RNA's reference sequence
  - Both axes are 1-based reference positions (1 to N)
  - For every read pair mapping to that RNA:
      * Parses the XD dot-bracket tag (full chimeric read structure with & as junction)
      * Finds cross-junction base pairs — ( ) positions that span across the &
      * Maps each paired position from chimeric-read-local coordinates to
        1-based reference coordinates using the CIGAR alignment
      * Scores +1 at matrix[arm1_ref_pos][arm2_ref_pos]
  - Only paired (bracketed) positions are scored — dots are ignored

Output per RNA:
  <rna>_contact_matrix.csv     N×N matrix, rows/cols labelled 1..N (1-based)
  <rna>_contact_matrix_sparse.tsv  sparse format: pos_i, pos_j, count (1-based)

Usage
─────
  conda activate bio
python build_contact_matrix.py \
  --bam   /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA/condition1_splits_filtered_sorted.bam \
  --fasta /mnt/ceph/mjain/software/envs/bio/custom_ref_genome/ecoli_rRNA/ecoli_rRNA_reference_dna.fa \
  --out   /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA
Dependencies: pysam, pandas, numpy
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pysam


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bam",   required=True,
                   help="Filtered splits BAM (condition1_splits_filtered_sorted.bam)")
    p.add_argument("--fasta", required=True,
                   help="Reference FASTA used for alignment (DNA alphabet)")
    p.add_argument("--out",   required=True,
                   help="Output directory (created if absent)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD REFERENCE SEQUENCE LENGTHS
# ══════════════════════════════════════════════════════════════════════════════

def load_reference_lengths(fasta_path: str) -> dict[str, int]:
    """Parse FASTA and return {sequence_name: length}."""
    lengths = {}
    name, length = None, 0
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    lengths[name] = length
                name   = line[1:].split()[0]
                length = 0
            else:
                length += len(line)
    if name:
        lengths[name] = length
    return lengths


# ══════════════════════════════════════════════════════════════════════════════
# PARSE XD DOT-BRACKET → CROSS-JUNCTION PAIRS
# ══════════════════════════════════════════════════════════════════════════════

def parse_cross_junction_pairs(xd: str) -> tuple[int, list[tuple[int, int]]]:
    """
    Parse the XD dot-bracket tag and return cross-junction paired positions.

    XD = structure of the full chimeric read with & as the junction marker:
         (((((((((...))))....&...........)))))....
         Left of &  = arm1 fragment structure
         Right of & = arm2 fragment structure

    A cross-junction pair has one position in arm1 (< &) and one in arm2 (>= &).
    The & character is NOT a structural element — it is removed before parsing
    but its position defines arm1_len.

    Returns:
      arm1_len  : number of arm1 characters before &
      cross     : list of (arm1_local_pos, arm2_local_pos) — both 0-based,
                  arm2_local_pos is relative to the start of arm2 (& removed)
    """
    amp_idx = xd.find("&")
    if amp_idx == -1:
        return 0, []

    arm1_len = amp_idx
    struct   = xd.replace("&", "")

    # Match all () pairs using a stack
    stack: list[int] = []
    all_pairs: list[tuple[int, int]] = []
    for i, c in enumerate(struct):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            all_pairs.append((j, i))   # (open, close) 0-based in chimeric read

    # Keep only cross-junction pairs
    cross = [
        (a, b - arm1_len)              # arm2_local = position relative to arm2 start
        for a, b in all_pairs
        if a < arm1_len <= b
    ]

    return arm1_len, cross


# ══════════════════════════════════════════════════════════════════════════════
# MAP LOCAL POSITIONS → REFERENCE POSITIONS
# ══════════════════════════════════════════════════════════════════════════════

def map_to_reference(
    cross:        list[tuple[int, int]],
    arm1_ref_pos: list[int],
    arm2_ref_pos: list[int],
) -> list[tuple[int, int]]:
    """
    Convert (arm1_local, arm2_local) cross-junction pairs to
    (arm1_ref_pos, arm2_ref_pos) in 1-based reference coordinates.

    arm1_ref_pos: list of 1-based reference positions for arm1 (from CIGAR)
    arm2_ref_pos: list of 1-based reference positions for arm2 (from CIGAR)

    Positions out of range (e.g. soft-clipped bases that have no ref coord)
    are silently skipped.
    """
    ref_pairs = []
    for a1_local, a2_local in cross:
        if a1_local >= len(arm1_ref_pos):
            continue
        if a2_local >= len(arm2_ref_pos):
            continue
        r1 = arm1_ref_pos[a1_local]
        r2 = arm2_ref_pos[a2_local]
        if r1 != r2:                   # same position pairing with itself = skip
            ref_pairs.append((r1, r2))
    return ref_pairs


# ══════════════════════════════════════════════════════════════════════════════
# BUILD CONTACT MATRICES FROM BAM
# ══════════════════════════════════════════════════════════════════════════════

def build_contact_matrices(
    bam_path: str,
    ref_lengths: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Iterate over the filtered BAM and accumulate contact scores.

    Each chimeric read appears as TWO consecutive records (same query_name):
      Record 1: arm1 alignment
      Record 2: arm2 alignment

    Returns:
      matrices[rna_name] = N×N numpy array (0-indexed internally, 1-based labels in output)
    """
    # Pre-allocate a matrix for every RNA in the reference
    matrices: dict[str, np.ndarray] = {
        name: np.zeros((length, length), dtype=np.int32)
        for name, length in ref_lengths.items()
    }

    stats = dict(
        total_records=0, pairs_processed=0,
        no_xd=0, out_of_range=0, contacts_scored=0
    )

    pending: dict[str, pysam.AlignedSegment] = {}

    bam = pysam.AlignmentFile(bam_path, "rb")

    for read in bam:
        stats["total_records"] += 1
        name = read.query_name

        if read.is_unmapped:
            pending.pop(name, None)
            continue

        if name not in pending:
            pending[name] = read
            continue

        # Second record — we now have both arms
        arm1_read = pending.pop(name)
        arm2_read = read
        stats["pairs_processed"] += 1

        rna_name = arm1_read.reference_name

        # Both arms must map to same RNA (should always be true in filtered BAM)
        if arm2_read.reference_name != rna_name:
            continue

        if rna_name not in matrices:
            continue

        # Get XD from arm1 record (both records carry the same XD)
        try:
            xd = arm1_read.get_tag("XD")
        except KeyError:
            stats["no_xd"] += 1
            continue

        # Parse cross-junction pairs from dot-bracket
        arm1_len, cross = parse_cross_junction_pairs(xd)
        if not cross:
            continue

        # 1-based reference positions for each arm via CIGAR
        arm1_ref_pos = [p + 1 for p in arm1_read.get_reference_positions()]
        arm2_ref_pos = [p + 1 for p in arm2_read.get_reference_positions()]

        if not arm1_ref_pos or not arm2_ref_pos:
            continue

        # Map to reference and score
        ref_pairs = map_to_reference(cross, arm1_ref_pos, arm2_ref_pos)
        rna_len   = ref_lengths[rna_name]

        for r1, r2 in ref_pairs:
            if not (1 <= r1 <= rna_len and 1 <= r2 <= rna_len):
                stats["out_of_range"] += 1
                continue
            # Score: arm1 on x-axis (column), arm2 on y-axis (row)
            # matrix is 0-indexed internally; labels are 1-based in output
            matrices[rna_name][r1 - 1][r2 - 1] += 1
            stats["contacts_scored"] += 1

    bam.close()

    print(f"\n  Processing summary:")
    print(f"    Total BAM records:       {stats['total_records']:>8,}")
    print(f"    Chimeric pairs processed:{stats['pairs_processed']:>8,}")
    print(f"    No XD tag (skipped):     {stats['no_xd']:>8,}")
    print(f"    Positions out of range:  {stats['out_of_range']:>8,}")
    print(f"    Contacts scored (+1):    {stats['contacts_scored']:>8,}")

    return matrices


# ══════════════════════════════════════════════════════════════════════════════
# WRITE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def write_outputs(
    rna_name: str,
    matrix:   np.ndarray,
    out_dir:  Path,
) -> None:
    n      = matrix.shape[0]
    labels = [str(i) for i in range(1, n + 1)]   # 1-based labels
    safe   = rna_name.replace("/", "_")

    # 1. Full N×N CSV — rows = arm1 positions (x), cols = arm2 positions (y)
    csv_path = out_dir / f"{safe}_contact_matrix.csv"
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(csv_path)
    print(f"    Matrix CSV:   {csv_path.name}  ({n}×{n})")

    # 2. Sparse TSV — only non-zero cells, easier to inspect and cross-reference
    rows, cols = np.nonzero(matrix)
    sparse_data = [
        {"pos_arm1": int(rows[i]) + 1,   # convert back to 1-based
         "pos_arm2": int(cols[i]) + 1,
         "count":    int(matrix[rows[i], cols[i]])}
        for i in range(len(rows))
    ]
    sparse_path = out_dir / f"{safe}_contact_matrix_sparse.tsv"
    if sparse_data:
        (pd.DataFrame(sparse_data)
           .sort_values("count", ascending=False)
           .to_csv(sparse_path, sep="\t", index=False))
    print(f"    Sparse TSV:   {sparse_path.name}  ({len(sparse_data)} non-zero cells)")

    # 3. Print top 10 most-observed pairs for quick sanity check
    top = sorted(sparse_data, key=lambda x: -x["count"])[:10]
    if top:
        print(f"    Top pairs (arm1_pos, arm2_pos, count):")
        for row in top:
            print(f"      pos {row['pos_arm1']:>5} ↔ pos {row['pos_arm2']:>5} : {row['count']:>4} reads")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Build Contact Matrices")
    print(f"{'='*60}")
    print(f"  BAM:    {args.bam}")
    print(f"  FASTA:  {args.fasta}")
    print(f"  Output: {args.out}")

    # Load reference lengths
    print(f"\n[1/3] Loading reference sequence lengths...")
    ref_lengths = load_reference_lengths(args.fasta)
    for name, length in ref_lengths.items():
        print(f"  {name.split('_URS')[0]}: {length} nt  →  {length}×{length} matrix")

    # Build matrices
    print(f"\n[2/3] Building contact matrices from BAM...")
    matrices = build_contact_matrices(args.bam, ref_lengths)

    # Write outputs
    print(f"\n[3/3] Writing outputs...")
    for rna_name, matrix in sorted(matrices.items()):
        total = int(matrix.sum())
        nonzero = int(np.count_nonzero(matrix))
        short = rna_name.split('_URS')[0]
        print(f"\n  [{short}]")
        print(f"    Total contacts scored:  {total:,}")
        print(f"    Non-zero matrix cells:  {nonzero:,}")
        if total == 0:
            print(f"    (no reads mapped to this RNA in filtered BAM — skipping output)")
            continue
        write_outputs(rna_name, matrix, out_dir)

    print(f"\n{'='*60}")
    print(f"  Done. Matrices written to: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()