#!/usr/bin/env python3
"""
filter_splits_bam.py
────────────────────
Filters condition1_splits.bam to keep only reads that:
  1. Appear in condition1_interactions_readIDs.tsv (survived RNAnue clustering)
  2. Are intramolecular (both arms map to the same reference sequence)

Output: a filtered BAM file with only those reads.

Usage
─────

python filter_splits_bam.py \
  --bam /mnt/ceph/mjain/software/envs/bio/rnanue_run/human_rRNA/output/03_detect/treatment/condition1/condition1_splits.bam \
  --ids /mnt/ceph/mjain/software/envs/bio/rnanue_run/human_rRNA/output/04_analyze/treatment/condition1/condition1_interactions_readIDs.tsv \
  --out /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/condition1_splits_filtered.bam
"""

import argparse
from pathlib import Path
import pysam


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bam", required=True,
                   help="condition1_splits.bam from RNAnue detect step")
    p.add_argument("--ids", required=True,
                   help="condition1_interactions_readIDs.tsv from analyze step")
    p.add_argument("--out", required=True,
                   help="Output filtered BAM path")
    return p.parse_args()


def load_valid_ids(tsv_path: str) -> set[str]:
    """
    Parse condition1_interactions_readIDs.tsv.
    Format: cluster_ID <TAB> comma-separated read IDs
    Returns flat set of all read IDs that survived clustering.
    """
    valid: set[str] = set()
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("cluster_ID"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            for rid in parts[1].split(","):
                rid = rid.strip()
                if rid:
                    valid.add(rid)
    print(f"  Valid read IDs loaded: {len(valid):,}")
    return valid


def filter_bam(bam_path: str, valid_ids: set[str], out_path: str) -> None:
    """
    Two-pass filter:
      Pass 1 — identify read names where BOTH arms are in valid_ids AND
               both arms map to the same reference (intramolecular).
      Pass 2 — write all records with those names to the output BAM.

    We need two passes because both records of a chimeric pair must pass
    before we commit either to the output.
    """

    # ── Pass 1: collect valid intramolecular read names ───────────────────────
    print("  Pass 1: identifying valid intramolecular read pairs...")

    bam_in = pysam.AlignmentFile(bam_path, "rb")

    # Buffer first record of each pair {query_name → reference_name}
    pending: dict[str, str] = {}

    kept_names:  set[str] = set()
    stats = dict(
        total=0, not_in_ids=0, intermolecular=0, kept=0
    )

    for read in bam_in:
        stats["total"] += 1
        name = read.query_name

        if read.is_unmapped:
            pending.pop(name, None)
            continue

        # Filter: must be in valid IDs
        if name not in valid_ids:
            stats["not_in_ids"] += 1
            pending.pop(name, None)
            continue

        if name not in pending:
            # First record of this pair — buffer it
            pending[name] = read.reference_name
        else:
            # Second record — now we can check intramolecular
            first_ref = pending.pop(name)
            second_ref = read.reference_name

            if first_ref != second_ref:
                stats["intermolecular"] += 1
                continue

            # Both filters passed — mark this name as kept
            kept_names.add(name)
            stats["kept"] += 1

    bam_in.close()

    print(f"    Total records:              {stats['total']:>8,}")
    print(f"    Not in valid IDs (skipped): {stats['not_in_ids']:>8,}")
    print(f"    Intermolecular (filtered):  {stats['intermolecular']:>8,}")
    print(f"    Pairs kept:                 {stats['kept']:>8,}")
    print(f"    Records to write:           {stats['kept']*2:>8,}  (2 per pair)")

    # ── Pass 2: write kept records to output BAM ──────────────────────────────
    print("  Pass 2: writing filtered BAM...")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    bam_in  = pysam.AlignmentFile(bam_path, "rb")
    bam_out = pysam.AlignmentFile(out_path, "wb", header=bam_in.header)

    written = 0
    for read in bam_in:
        if read.query_name in kept_names:
            bam_out.write(read)
            written += 1

    bam_in.close()
    bam_out.close()

    print(f"    Records written: {written:,}")

    # ── Index the output BAM ──────────────────────────────────────────────────
    print("  Sorting and indexing output BAM...")
    sorted_path = out_path.replace(".bam", "_sorted.bam")
    pysam.sort("-o", sorted_path, out_path)
    pysam.index(sorted_path)
    print(f"  Sorted BAM:  {sorted_path}")
    print(f"  Index:       {sorted_path}.bai")


def main():
    args = parse_args()

    print(f"\n{'='*55}")
    print(f"  Filter splits BAM")
    print(f"{'='*55}")
    print(f"  BAM:    {args.bam}")
    print(f"  IDs:    {args.ids}")
    print(f"  Output: {args.out}")
    print()

    print("[1/2] Loading valid read IDs...")
    valid_ids = load_valid_ids(args.ids)

    print("\n[2/2] Filtering BAM...")
    filter_bam(args.bam, valid_ids, args.out)

    print(f"\n{'='*55}")
    print(f"  Done.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()