#!/usr/bin/env python3
"""
compare_rnafold.py
───────────────────
Compares RNAfold predicted structures (constrained and unconstrained) against
the known ground truth structure from the rRNA/tRNA CSV file.

Evaluation is at the exact base-pair level:
  TP = base pair in both predicted and ground truth
  FP = base pair in predicted but not ground truth
  FN = base pair in ground truth but not predicted
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 * P * R / (P + R)

Usage
─────
python compare_rnafold.py --rna_csv /mnt/ceph/mjain/software/envs/bio/custom_ref_genome/ecoli_rRNA/ecoli_rRNA.csv --constrained /mnt/ceph/mjain/software/envs/bio/rnafold_output/ecoli_rRNA/all_rnafold_constrained_output.txt --unconstrained /mnt/ceph/mjain/software/envs/bio/rnafold_output/ecoli_rRNA/all_rnafold_unconstrained_output.txt --out /mnt/ceph/mjain/software/envs/bio/rnafold_comparison/ecoli_rRNA

Both --constrained and --unconstrained point to the combined output files
(all RNAs in one file, produced by run_rnafold.sh). The script parses each
file by matching the >header line to the RNA names in --rna_csv.

Output:
  results_summary.tsv    — one row per RNA: metrics for both conditions
  <rna>_comparison.txt   — detailed per-RNA breakdown

Dependencies: pandas
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rna_csv",       required=True,
                   help="CSV with name, seq, sec_str columns (ground truth)")
    p.add_argument("--constrained",   required=True,
                   help="Combined RNAfold constrained output file (all_rnafold_constrained_output.txt)")
    p.add_argument("--unconstrained", required=True,
                   help="Combined RNAfold unconstrained output file (all_rnafold_unconstrained_output.txt)")
    p.add_argument("--out",           required=True,
                   help="Output directory")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# BASE PAIR PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_pairs(struct: str) -> set[tuple[int, int]]:
    """
    Parse a dot-bracket structure and return a set of (i, j) base pairs.
    i < j always. Positions are 1-based.
    Handles standard ( ) brackets only — ignores pseudoknot brackets.
    """
    struct = struct.strip().split()[0]  # remove trailing free energy if present
    stack: list[int] = []
    pairs: set[tuple[int, int]] = set()
    for idx, c in enumerate(struct):
        if c == "(":
            stack.append(idx + 1)
        elif c == ")" and stack:
            j = stack.pop()
            pairs.add((j, idx + 1))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# RNAFOLD OUTPUT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_rnafold_combined(file_path: Path) -> dict[str, tuple[str, float]]:
    """
    Parse a combined RNAfold output file containing multiple RNA results.

    Each entry has the format:
      >rna_name
      SEQUENCE
      STRUCTURE  (free_energy)

    Returns {rna_name: (structure_string, free_energy)} for all entries.
    Also stores results under the short name (before _URS) as a fallback key.
    """
    results: dict[str, tuple[str, float]] = {}
    if not file_path.exists():
        return results

    lines = [l.rstrip() for l in file_path.read_text().splitlines()]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if not lines[i].startswith(">"):
            i += 1
            continue

        rna_name = lines[i].lstrip(">").split()[0]
        i += 1

        # Skip sequence line
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        i += 1

        # Structure + energy line
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        struct_line = lines[i].strip()
        i += 1

        energy = float("nan")
        m = re.search(r"\(\s*(-?\d+\.\d+)\s*\)", struct_line)
        if m:
            energy = float(m.group(1))

        structure = struct_line.split()[0]
        results[rna_name] = (structure, energy)
        # Also index by short name for fallback matching
        short = rna_name.split("_URS")[0]
        if short not in results:
            results[short] = (structure, energy)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    predicted:    set[tuple[int, int]],
    ground_truth: set[tuple[int, int]],
) -> dict:
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(
        tp=tp, fp=fp, fn=fn,
        n_predicted=len(predicted),
        n_ground_truth=len(ground_truth),
        precision=precision,
        recall=recall,
        f1=f1,
    )





# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    con_file   = Path(args.constrained)
    uncon_file = Path(args.unconstrained)

    print(f"\n{'='*60}")
    print(f"  RNAfold Structure Comparison")
    print(f"{'='*60}")
    print(f"  Constrained output:   {con_file.name}")
    print(f"  Unconstrained output: {uncon_file.name}")

    # Load ground truth
    rna_df = pd.read_csv(args.rna_csv)
    required = {"name", "sec_str", "seq"}
    if not required.issubset(rna_df.columns):
        print(f"ERROR: rna_csv must have columns: {required}")
        return

    # Parse both combined output files up front
    print(f"\nParsing RNAfold output files...")
    con_results   = parse_rnafold_combined(con_file)
    uncon_results = parse_rnafold_combined(uncon_file)
    print(f"  Constrained:   {len([k for k in con_results if '_URS' in k or len(k)<20])} RNAs found")
    print(f"  Unconstrained: {len([k for k in uncon_results if '_URS' in k or len(k)<20])} RNAs found")

    all_results = []

    for _, row in rna_df.iterrows():
        rna_name  = str(row["name"]).strip()
        sec_str   = str(row["sec_str"]).strip()
        short     = rna_name.split("_URS")[0] if "_URS" in rna_name else rna_name

        gt_pairs  = parse_pairs(sec_str)
        if not gt_pairs:
            print(f"\n  [{short}] WARNING: no base pairs parsed from sec_str — skipping")
            continue

        print(f"\n  [{short}]")
        print(f"    Ground truth base pairs: {len(gt_pairs)}")

        # Look up this RNA in the combined results by full name, then short name
        def lookup(results_dict, name, short):
            if name in results_dict:
                return results_dict[name]
            if short in results_dict:
                return results_dict[short]
            return None

        con_result   = lookup(con_results,   rna_name, short)
        uncon_result = lookup(uncon_results, rna_name, short)

        if not con_result:
            print(f"    WARNING: {short} not found in constrained output "
                  f"(may have been excluded due to no pairs above min_support)")
        if not uncon_result:
            print(f"    WARNING: {short} not found in unconstrained output")

        row_data = {
            "rna":                rna_name,
            "seq_len":            len(str(row["seq"]).strip()),
            "gt_pairs":           len(gt_pairs),
        }

        for label, result in [("constrained", con_result), ("unconstrained", uncon_result)]:
            if result is None:
                row_data.update({
                    f"{label}_pairs":     None,
                    f"{label}_energy":    None,
                    f"{label}_tp":        None,
                    f"{label}_fp":        None,
                    f"{label}_fn":        None,
                    f"{label}_precision": None,
                    f"{label}_recall":    None,
                    f"{label}_f1":        None,
                })
                continue

            structure, energy = result
            pred_pairs = parse_pairs(structure)
            metrics    = compute_metrics(pred_pairs, gt_pairs)

            print(f"    {label.capitalize()}: {len(pred_pairs)} pairs, "
                  f"energy={energy:.2f} kcal/mol")
            print(f"      TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
            print(f"      Precision={metrics['precision']:.1%}, "
                  f"Recall={metrics['recall']:.1%}, "
                  f"F1={metrics['f1']:.1%}")

            row_data.update({
                f"{label}_pairs":     len(pred_pairs),
                f"{label}_energy":    energy,
                f"{label}_tp":        metrics["tp"],
                f"{label}_fp":        metrics["fp"],
                f"{label}_fn":        metrics["fn"],
                f"{label}_precision": round(metrics["precision"], 4),
                f"{label}_recall":    round(metrics["recall"],    4),
                f"{label}_f1":        round(metrics["f1"],        4),
            })

        # ── Recall delta ─────────────────────────────────────────────────────
        con_r   = row_data.get("constrained_recall")
        uncon_r = row_data.get("unconstrained_recall")
        row_data["recall_delta"] = (
            round(con_r - uncon_r, 4)
            if con_r is not None and uncon_r is not None
            else None
        )

        # ── Improvement summary ───────────────────────────────────────────────
        if con_result and uncon_result:
            con_struct,   con_energy   = con_result
            uncon_struct, uncon_energy = uncon_result
            con_pairs   = parse_pairs(con_struct)
            uncon_pairs = parse_pairs(uncon_struct)
            con_m       = compute_metrics(con_pairs,   gt_pairs)
            uncon_m     = compute_metrics(uncon_pairs, gt_pairs)

            f1_delta = con_m["f1"] - uncon_m["f1"]
            print(f"    F1 change (constrained vs unconstrained): "
                  f"{f1_delta:+.1%}  "
                  f"{'↑ improved' if f1_delta > 0 else '↓ worse' if f1_delta < 0 else '= same'}")
            print(f"    Energy cost of constraints: "
                  f"{con_energy - uncon_energy:+.2f} kcal/mol")

            # ── Detailed per-RNA text report ──────────────────────────────────
            detail_path = out_dir / f"{short}_comparison.txt"
            with open(detail_path, "w") as f:
                f.write(f"{'='*55}\n")
                f.write(f"  {short}  ({row_data['seq_len']} nt)\n")
                f.write(f"{'='*55}\n\n")
                f.write(f"Ground truth base pairs: {len(gt_pairs)}\n\n")

                for lbl, struct, pairs, m, energy in [
                    ("Constrained",   con_struct,   con_pairs,   con_m,   con_energy),
                    ("Unconstrained", uncon_struct, uncon_pairs, uncon_m, uncon_energy),
                ]:
                    f.write(f"{lbl}:\n")
                    f.write(f"  Structure: {struct}\n")
                    f.write(f"  Energy:    {energy:.2f} kcal/mol\n")
                    f.write(f"  Pairs:     {len(pairs)}\n")
                    f.write(f"  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}\n")
                    f.write(f"  Precision={m['precision']:.1%}, "
                            f"Recall={m['recall']:.1%}, F1={m['f1']:.1%}\n\n")

                f.write(f"Improvement from DDD constraints:\n")
                f.write(f"  F1:        {f1_delta:+.1%}\n")
                f.write(f"  Precision: {con_m['precision']-uncon_m['precision']:+.1%}\n")
                f.write(f"  Recall:    {con_m['recall']-uncon_m['recall']:+.1%}\n")
                f.write(f"  Energy cost of constraints: "
                        f"{con_energy-uncon_energy:+.2f} kcal/mol\n\n")

                f.write(f"Pairs only in constrained (new correct pairs added):\n")
                gained = (con_pairs & gt_pairs) - (uncon_pairs & gt_pairs)
                for pair in sorted(gained):
                    f.write(f"  {pair}\n")
                if not gained:
                    f.write(f"  (none)\n")

                f.write(f"\nPairs only in unconstrained (correct pairs lost):\n")
                lost = (uncon_pairs & gt_pairs) - (con_pairs & gt_pairs)
                for pair in sorted(lost):
                    f.write(f"  {pair}\n")
                if not lost:
                    f.write(f"  (none)\n")

            print(f"    Detail report → {detail_path.name}")

        all_results.append(row_data)

    # ── Summary TSV ───────────────────────────────────────────────────────────
    if all_results:
        summary_path = out_dir / "results_summary.tsv"
        df = pd.DataFrame(all_results)
        # Insert recall_delta immediately after constrained_pairs
        cols = list(df.columns)
        cols.insert(cols.index("constrained_pairs") + 1, cols.pop(cols.index("recall_delta")))
        df[cols].to_csv(summary_path, sep="\t", index=False)
        print(f"\n  Summary TSV → {summary_path}")

    print(f"\n{'='*60}")
    print(f"  Done. Results in: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()