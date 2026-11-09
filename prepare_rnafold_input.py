#!/usr/bin/env python3
"""
prepare_rnafold_input.py
─────────────────────────
Generates RNAfold hard constraint input files from contact matrix sparse TSVs.

From the contact matrix, extracts high-confidence base-pair contacts and
converts them into an RNAfold hard constraint string. Also writes the
corresponding RNA FASTA file with U (not T) alphabet.

RNAfold hard constraint format:
  >rna_name
  RNASEQ (U alphabet)
  constraint_string

Constraint string characters (same length as sequence):
  .  unconstrained — RNAfold decides freely
  (  forced to pair with the matching ) downstream
  )  forced to pair with the matching ( upstream

Conflict resolution (required before passing to RNAfold):
  1. Multi-partner: if position i appears in multiple pairs,
     keep only the highest-count pair for that position.
  2. Pseudoknot removal: RNAfold does not support pseudoknots.
     Crossing pairs are resolved by keeping higher-support pair.
     Removed pairs are logged for inspection.

Usage
─────
  conda activate bio
  python prepare_rnafold_input.py \
      --sparse_dir  ./contact_matrices/ecoli_rRNA \
      --fasta       .../ecoli_rRNA_reference_dna.fa \
      --out         ./rnafold_input/ecoli_rRNA \
      --min_support 20

  Then run RNAfold:
      RNAfold --constraint --noPS < <rna>_rnafold_input.txt

Dependencies: pandas, numpy
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sparse_dir",  required=True)
    p.add_argument("--fasta",       required=True)
    p.add_argument("--out",         required=True)
    p.add_argument("--min_support", type=int, default=20,
                   help="Min read count for a pair to become a hard constraint (default 20)")
    return p.parse_args()


def load_reference(fasta_path):
    seqs = {}
    name, buf = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf).upper()
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name:
        seqs[name] = "".join(buf).upper()
    return seqs


def resolve_multi_partner(pairs):
    """Keep only the highest-count pair for each position."""
    best = {}
    for i, j, c in sorted(pairs, key=lambda x: -x[2]):
        if i not in best and j not in best:
            best[i] = (j, c)
            best[j] = (i, c)
    seen, resolved = set(), []
    for pos, (partner, count) in best.items():
        key = (min(pos, partner), max(pos, partner))
        if key not in seen:
            seen.add(key)
            resolved.append(key + (count,))
    return sorted(resolved)


def remove_pseudoknots(pairs):
    """Remove crossing pairs, keeping higher-support pair in each conflict."""
    sorted_pairs = sorted(pairs, key=lambda x: (-x[2], x[1] - x[0]))
    kept, removed = [], []
    for candidate in sorted_pairs:
        i, j, c = candidate
        crosses = any(
            (ki < i < kj < j) or (i < ki < j < kj)
            for ki, kj, _ in kept
        )
        (removed if crosses else kept).append(candidate)
    return kept, removed


def build_constraint(pairs, seq_len):
    """Build RNAfold constraint string. pos_i gets '(', pos_j gets ')'."""
    constraint = ["."] * seq_len
    for i, j, _ in pairs:
        if 1 <= i <= seq_len and 1 <= j <= seq_len:
            constraint[i - 1] = "("
            constraint[j - 1] = ")"
    return "".join(constraint)


def match_rna_name(sparse_stem, ref_seqs):
    candidate = sparse_stem.replace("_contact_matrix_sparse", "")
    if candidate in ref_seqs:
        return candidate
    short = candidate.split("_URS")[0]
    for key in ref_seqs:
        if key.split("_URS")[0] == short:
            return key
    return None


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Prepare RNAfold Hard Constraint Input")
    print(f"{'='*60}")
    print(f"  Sparse dir:  {args.sparse_dir}")
    print(f"  FASTA:       {args.fasta}")
    print(f"  Min support: {args.min_support}")

    ref_seqs = load_reference(args.fasta)
    print(f"\nReference sequences loaded:")
    for name, seq in ref_seqs.items():
        print(f"  {name.split('_URS')[0]}: {len(seq)} nt")

    sparse_files = sorted(Path(args.sparse_dir).glob("*_contact_matrix_sparse.tsv"))
    if not sparse_files:
        print(f"ERROR: No sparse TSV files found in {args.sparse_dir}")
        return

    run_cmds = []

    for sp_path in sparse_files:
        rna_name = match_rna_name(sp_path.stem, ref_seqs)
        if rna_name is None:
            print(f"\nWARNING: Cannot match {sp_path.name} to reference — skipping")
            continue

        short   = rna_name.split("_URS")[0]
        rna_seq = ref_seqs[rna_name]
        seq_len = len(rna_seq)

        print(f"\n[{short}]  ({seq_len} nt)")

        df = pd.read_csv(sp_path, sep="\t")
        hi = df[df["count"] >= args.min_support]
        print(f"  Pairs >= {args.min_support} reads: {len(hi)}")

        if hi.empty:
            print(f"  No pairs above threshold — lower --min_support to include more")
            continue

        # Normalise: ensure pos_i < pos_j
        raw = [(min(int(r.pos_arm1), int(r.pos_arm2)),
                max(int(r.pos_arm1), int(r.pos_arm2)),
                int(r['count']))
               for _, r in hi.iterrows()
               if int(r.pos_arm1) != int(r.pos_arm2)]

        # Resolve conflicts
        after_mp = resolve_multi_partner(raw)
        removed_mp_n = len(raw) - len(after_mp)
        print(f"  After multi-partner resolution: {len(after_mp)} (removed {removed_mp_n})")

        kept, removed_pk = remove_pseudoknots(after_mp)
        print(f"  After pseudoknot removal:       {len(kept)} (removed {len(removed_pk)})")

        if not kept:
            print(f"  No valid pairs remain")
            continue

        # Build constraint string
        constraint = build_constraint(kept, seq_len)
        n_constrained = sum(1 for c in constraint if c != ".")
        print(f"  Constrained positions: {n_constrained}/{seq_len}")

        # Show pairs
        print(f"  Pairs used as hard constraints:")
        for i, j, c in kept:
            print(f"    pos {i:>5} ( ↔ pos {j:>5} )  count={c}")

        # ── Write RNA FASTA (U alphabet) ──────────────────────────────────────
        safe     = rna_name.replace("/", "_")
        rna_u    = rna_seq.replace("T", "U")

        fasta_path = out_dir / f"{safe}_rna.fa"
        with open(fasta_path, "w") as f:
            f.write(f">{rna_name}\n")
            for i in range(0, len(rna_u), 60):
                f.write(rna_u[i:i+60] + "\n")

        # ── Write RNAfold input: header + sequence + constraint ───────────────
        input_path = out_dir / f"{safe}_rnafold_input.txt"
        with open(input_path, "w") as f:
            f.write(f">{rna_name}\n")
            f.write(f"{rna_u}\n")
            f.write(f"{constraint}\n")

        # ── Write constraint pairs log ────────────────────────────────────────
        log_rows = (
            [{"pos_i": i, "pos_j": j, "count": c, "status": "KEPT"}
             for i, j, c in kept] +
            [{"pos_i": i, "pos_j": j, "count": c, "status": "REMOVED_multi_partner"}
             for i, j, c in raw if (min(i,j), max(i,j), c) not in [(a,b,cc) for a,b,cc in after_mp]] +
            [{"pos_i": i, "pos_j": j, "count": c, "status": "REMOVED_pseudoknot"}
             for i, j, c in removed_pk]
        )
        log_path = out_dir / f"{safe}_constraint_pairs.tsv"
        pd.DataFrame(log_rows).sort_values("pos_i").to_csv(log_path, sep="\t", index=False)

        print(f"  → RNA FASTA:      {fasta_path.name}")
        print(f"  → RNAfold input:  {input_path.name}")
        print(f"  → Pairs log:      {log_path.name}")

        run_cmds.append((short, safe))

    # ── Write run script ──────────────────────────────────────────────────────
    sh_path = out_dir / "run_rnafold.sh"
    with open(sh_path, "w") as f:
        f.write("#!/bin/bash\n# Run RNAfold with hard constraints\n\n")
        for short, safe in run_cmds:
            f.write(f"# {short}\n")
            f.write(f"RNAfold --constraint --noPS "
                    f"< {out_dir}/{safe}_rnafold_input.txt "
                    f"> {out_dir}/{safe}_rnafold_output.txt\n\n")
    print(f"\nRun script: {sh_path}")
    print(f"\nTo run all:\n  bash {sh_path}")
    print(f"\nOr one at a time:\n  RNAfold --constraint --noPS < <rna>_rnafold_input.txt")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()