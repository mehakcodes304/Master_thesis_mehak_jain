#!/usr/bin/env python3
"""
make_zero_reads_report.py — reconstruct a zero_reads_report.tsv from existing
chimeric reads CSV files WITHOUT re-running generation.

Compares every (name, interaction_number, orientation) expected from the stems
CSV against what actually appears in the chimeric reads CSV.  Any combo that is
missing from the reads CSV = 0 reads generated → written to the report.

Usage
-----
    python make_zero_reads_report.py \
        --stems  stems2_ecoli_tRNA.csv \
        --reads  ecoli_tRNA_chimeric_reads1.csv \
        --out    ecoli_tRNA_chimeric_reads1_zero_reads_report.tsv

The output TSV has three columns: name, interaction_number, orientation
and is directly usable by filter_stems.py.
"""

import argparse
import pandas as pd
from pathlib import Path

ORIENTATIONS = ["arm1_to_arm2", "arm2_to_arm1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True, help="stems2_*.csv")
    ap.add_argument("--reads", required=True, help="chimeric reads CSV")
    ap.add_argument("--out",   default=None,  help="output TSV path (default: auto-named next to --reads)")
    args = ap.parse_args()

    stems_path = Path(args.stems)
    reads_path = Path(args.reads)

    if args.out is None:
        out_path = reads_path.parent / f"{reads_path.stem}_zero_reads_report.tsv"
    else:
        out_path = Path(args.out)

    stems = pd.read_csv(stems_path)
    reads = pd.read_csv(reads_path, usecols=["name", "interaction_number", "orientation"])

    print(f"Stems CSV:   {stems_path}  ({len(stems)} rows, "
          f"{stems[['name','interaction_number']].drop_duplicates().shape[0]} unique interactions)")
    print(f"Reads CSV:   {reads_path}  ({len(reads)} reads)")

    # All expected combos: each (name, interaction_number) × both orientations
    expected_interactions = stems[["name", "interaction_number"]].drop_duplicates()
    expected_rows = []
    for _, row in expected_interactions.iterrows():
        for ori in ORIENTATIONS:
            expected_rows.append({"name": row["name"],
                                   "interaction_number": int(row["interaction_number"]),
                                   "orientation": ori})
    expected = pd.DataFrame(expected_rows)

    # What actually exists in the reads CSV
    present = (
        reads[["name", "interaction_number", "orientation"]]
        .drop_duplicates()
        .assign(_present=True)
    )

    # Left join — missing ones have NaN in _present
    merged = expected.merge(present, on=["name", "interaction_number", "orientation"], how="left")
    zero = merged[merged["_present"].isna()][["name", "interaction_number", "orientation"]]

    print(f"\nExpected combos:  {len(expected)}")
    print(f"Present in reads: {len(present)}")
    print(f"Zero-read combos: {len(zero)}")

    if not zero.empty:
        print("\nZero-read interactions:")
        for _, row in zero.iterrows():
            print(f"  {row['name']}  int {int(row['interaction_number'])}  {row['orientation']}")

    zero.to_csv(out_path, sep="\t", index=False)
    print(f"\nReport written: {out_path}")
    print("\nNext step:")
    print(f"  python filter_stems.py --stems {stems_path} --report {out_path}")


if __name__ == "__main__":
    main()