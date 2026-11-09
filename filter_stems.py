#!/usr/bin/env python3
"""
filter_stems.py — produce a filtered stems2 CSV that removes interactions
which generated 0 chimeric reads during simulation.

An interaction (name, interaction_number) is removed from the GT if BOTH
arm1_to_arm2 AND arm2_to_arm1 had 0 reads.  If only one direction failed,
the interaction is still kept (the other direction did generate reads, so
RNAfold will still receive constraints for it).

Usage
-----
    python filter_stems.py \
        --stems  stems2_ecoli_tRNA.csv \
        --report ecoli_tRNA_chimeric_reads_zero_reads_report.tsv \
        --out    stems2_ecoli_tRNA_filtered.csv

    # or let the script infer the output name:
    python filter_stems.py \
        --stems  stems2_ecoli_tRNA.csv \
        --report ecoli_tRNA_chimeric_reads_zero_reads_report.tsv
"""

import argparse
import pandas as pd
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter stems2 CSV by zero-read interactions.")
    ap.add_argument("--stems",  required=True, help="Path to stems2_*.csv")
    ap.add_argument("--report", required=True, help="Path to *_zero_reads_report.tsv")
    ap.add_argument("--out",    default=None,
                    help="Output path (default: stems2_*_filtered.csv next to --stems)")
    args = ap.parse_args()

    stems_path  = Path(args.stems)
    report_path = Path(args.report)

    if args.out is None:
        stem_name = stems_path.stem          # e.g. stems2_ecoli_tRNA
        out_path  = stems_path.parent / f"{stem_name}_filtered.csv"
    else:
        out_path  = Path(args.out)

    # Load inputs
    stems  = pd.read_csv(stems_path)
    report = pd.read_csv(report_path, sep="\t")

    print(f"Stems CSV:    {stems_path}  ({len(stems)} rows)")
    print(f"Zero report:  {report_path}  ({len(report)} zero-read entries)")

    if report.empty:
        print("No zero-read interactions — nothing to filter. Copying stems CSV as-is.")
        stems.to_csv(out_path, index=False)
        print(f"Written:      {out_path}")
        return

    # Count how many directions failed per (name, interaction_number)
    fail_counts = (
        report.groupby(["name", "interaction_number"])
              .size()
              .reset_index(name="failed_directions")
    )

    # An interaction has 2 possible directions; remove only if both failed
    both_failed = fail_counts[fail_counts["failed_directions"] >= 2][
        ["name", "interaction_number"]
    ]

    if both_failed.empty:
        print("No interactions had BOTH directions fail — nothing removed from GT.")
        stems.to_csv(out_path, index=False)
        print(f"Written:      {out_path}")
        return

    print(f"\nInteractions removed (both directions had 0 reads): {len(both_failed)}")
    for _, row in both_failed.iterrows():
        print(f"  {row['name']}  int {row['interaction_number']}")

    # Filter stems CSV
    merged = stems.merge(
        both_failed.assign(_drop=True),
        on=["name", "interaction_number"],
        how="left",
    )
    filtered = merged[merged["_drop"].isna()].drop(columns="_drop")

    removed = len(stems) - len(filtered)
    print(f"\nRows removed: {removed}  ({len(stems)} → {len(filtered)})")

    filtered.to_csv(out_path, index=False)
    print(f"Written:      {out_path}")


if __name__ == "__main__":
    main()