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

Pair selection uses a fraction-based threshold:
  For each position i, compute its total contacts across all partners.
  The best single partner j is the one with the highest combined contact count
  (matrix[i][j] + matrix[j][i]). If that fraction >= --pair_fraction, the pair
  (i, j) is a candidate constraint. A --min_count floor filters out positions
  with very few total contacts where a single noise read could give 100% fraction.

  Example: position A has 100 total contacts, 90 with position B → fraction=0.90
  With --pair_fraction 0.80, this pair is included. If A had only 3 total contacts
  (2 with B), fraction=0.67 < 0.80 and the pair is excluded.

Conflict resolution (required before passing to RNAfold):
  1. Multi-partner: fraction selection already yields at most one partner per
     position, but the mutual check may still leave conflicts — resolved by
     keeping the pair with the higher total contact count.
  2. Pseudoknot removal: RNAfold does not support pseudoknots.
     Crossing pairs are resolved by keeping higher-support pair.
     Removed pairs are logged for inspection.

Usage
─────
  conda activate bio
  python prepare_rnafold_input1.py \
      --sparse_dir  /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA \
      --fasta       /mnt/ceph/mjain/software/envs/bio/custom_ref_genome/ecoli_rRNA/ecoli_rRNA_reference_dna.fa \
      --out         /mnt/ceph/mjain/software/envs/bio/rnafold_input/ecoli_rRNA \
      --pair_fraction 0.50 \
      --min_count     5

  Then run RNAfold:
      RNAfold --constraint --noPS < all_rnafold_constrained_input.txt

OUT=/mnt/ceph/mjain/software/envs/bio/rnafold_input/human_tRNA

python prepare_rnafold_input1.py \
    --sparse_dir  /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_tRNA \
    --fasta       /mnt/ceph/mjain/software/envs/bio/custom_ref_genome/human_tRNA/human_tRNA_reference_dna.fa \
    --out         "$OUT" \
    --pair_fraction 0.50 \
    --min_count    3

# Move per-tRNA files into their subdirectories
for f in "$OUT"/tRNA_*.fa "$OUT"/tRNA_*_constraint_pairs.tsv; do
    [ -f "$f" ] || continue
    name=$(basename "$f" | sed 's/_URS[A-Z0-9]*_[0-9]*//; s/_rna\.fa$//; s/_constraint_pairs\.tsv$//')
    mkdir -p "$OUT/$name"
    mv "$f" "$OUT/$name/"
done
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sparse_dir",  required=True)
    p.add_argument("--fasta",       required=True)
    p.add_argument("--out",         required=True)
    p.add_argument("--pair_fraction", type=float,
                   help="Min fraction of a position's total contacts that must go "
                        "to one partner for that pair to become a constraint ")
    p.add_argument("--min_count",    type=int,
                   help="Min total contact count for a position to be considered "
                        "at all — prevents noise from positions seen in only 1-2 reads ")
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


def select_pairs_by_fraction(
    sparse_df: pd.DataFrame,
    seq_len:   int,
    pair_fraction: float,
    min_count: int,
) -> list[tuple[int, int, float]]:
    """
    Select base-pair candidates from the sparse contact matrix using a
    fraction-based threshold rather than an absolute count threshold.

    For each position p (1-based):
      1. Sum all contacts involving p (as arm1 OR arm2): total_p
      2. If total_p < min_count: skip (too few contacts — likely noise)
      3. Find the best partner j: highest combined count matrix[p][j] + matrix[j][p]
      4. fraction = best_combined / total_p
      5. If fraction >= pair_fraction: add (min(p,j), max(p,j)) as a candidate

    Returns list of (pos_i, pos_j, combined_count) with pos_i < pos_j.
    Duplicates are deduplicated — the same pair found from both directions
    is counted only once.
    """
    # Build per-position contact dictionaries from sparse TSV
    # contacts_as_arm1[p][j] = count where p was arm1, j was arm2
    # contacts_as_arm2[p][j] = count where p was arm2, j was arm1
    from collections import defaultdict
    as_arm1: dict = defaultdict(dict)
    as_arm2: dict = defaultdict(dict)

    for _, row in sparse_df.iterrows():
        i, j, c = int(row["pos_arm1"]), int(row["pos_arm2"]), int(row["count"])
        if i == j:
            continue
        as_arm1[i][j] = as_arm1[i].get(j, 0) + c
        as_arm2[j][i] = as_arm2[j].get(i, 0) + c

    # All positions that appear in at least one contact
    all_positions = set(as_arm1.keys()) | set(as_arm2.keys())

    candidates: set[tuple[int, int]] = set()
    candidate_counts: dict[tuple, float] = {}

    for p in all_positions:
        # Total contacts for this position
        total_p = (sum(as_arm1[p].values()) if p in as_arm1 else 0) + \
                  (sum(as_arm2[p].values()) if p in as_arm2 else 0)

        if total_p < min_count:
            continue

        # All partners and their combined counts (both directions)
        all_partners: dict[int, int] = {}
        for j, c in (as_arm1.get(p, {}) | {}).items():
            all_partners[j] = all_partners.get(j, 0) + c
        for j, c in (as_arm2.get(p, {}) | {}).items():
            all_partners[j] = all_partners.get(j, 0) + c

        if not all_partners:
            continue

        best_j     = max(all_partners, key=all_partners.get)
        best_count = all_partners[best_j]
        fraction   = best_count / total_p

        if fraction >= pair_fraction:
            key = (min(p, best_j), max(p, best_j))
            # Keep the entry with the higher combined count if already seen
            if key not in candidate_counts or best_count > candidate_counts[key]:
                candidate_counts[key] = best_count
                candidates.add(key)

    return sorted(
        [(i, j, candidate_counts[(i,j)]) for i,j in candidates],
        key=lambda x: x[0]
    )


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
    print(f"  Pair fraction: {args.pair_fraction:.0%}  (min fraction of contacts to one partner)")
    print(f"  Min total:     {args.min_count} contacts per position")

    ref_seqs = load_reference(args.fasta)
    print(f"\nReference sequences loaded:")
    for name, seq in ref_seqs.items():
        print(f"  {name.split('_URS')[0]}: {len(seq)} nt")

    sparse_files = sorted(Path(args.sparse_dir).glob("*_contact_matrix_sparse.tsv"))
    if not sparse_files:
        print(f"ERROR: No sparse TSV files found in {args.sparse_dir}")
        return

    # Collect entries for the two combined input files.
    # RNAfold reads them sequentially — 3 lines per entry for constrained
    # (header, sequence, constraint), 2 lines for unconstrained (header, sequence).
    # Per-RNA FASTA and constraint log files are still written separately
    # so individual RNAs can be inspected without regenerating everything.
    combined_con   = []   # list of (rna_name, rna_u, constraint)
    combined_uncon = []   # list of (rna_name, rna_u)

    for sp_path in sparse_files:
        rna_name = match_rna_name(sp_path.stem, ref_seqs)
        if rna_name is None:
            print(f"\nWARNING: Cannot match {sp_path.name} to reference — skipping")
            continue

        short   = rna_name.split("_URS")[0]
        rna_seq = ref_seqs[rna_name]
        seq_len = len(rna_seq)
        rna_u   = rna_seq.replace("T", "U")
        safe    = rna_name.replace("/", "_")

        print(f"\n[{short}]  ({seq_len} nt)")

        # Always include in unconstrained run regardless of pair support
        combined_uncon.append((rna_name, rna_u))

        df = pd.read_csv(sp_path, sep="\t")

        # Fraction-based pair selection
        raw = select_pairs_by_fraction(
            df, seq_len, args.pair_fraction, args.min_count
        )
        print(f"  Pairs selected (fraction>={args.pair_fraction:.0%}, "
              f"min_count>={args.min_count}): {len(raw)}")

        if not raw:
            print(f"  No pairs above threshold — RNA excluded from constrained run")
            continue

        # Resolve conflicts
        after_mp     = resolve_multi_partner(raw)
        removed_mp_n = len(raw) - len(after_mp)
        print(f"  After multi-partner resolution: {len(after_mp)} (removed {removed_mp_n})")

        kept, removed_pk = remove_pseudoknots(after_mp)
        print(f"  After pseudoknot removal:       {len(kept)} (removed {len(removed_pk)})")

        if not kept:
            print(f"  No valid pairs remain — RNA excluded from constrained run")
            continue

        # Build constraint string
        constraint    = build_constraint(kept, seq_len)
        n_constrained = sum(1 for c in constraint if c != ".")
        print(f"  Constrained positions: {n_constrained}/{seq_len}")

        print(f"  Pairs used as hard constraints:")
        for i, j, c in sorted(kept, key=lambda x: x[0]):
            print(f"    pos {i:>5} ( ↔ pos {j:>5} )  count={c}")

        # ── Per-RNA RNA FASTA (U alphabet) ────────────────────────────────────
        fasta_path = out_dir / f"{safe}_rna.fa"
        with open(fasta_path, "w") as f:
            f.write(f">{rna_name}\n")
            for i in range(0, len(rna_u), 60):
                f.write(rna_u[i:i+60] + "\n")

        # ── Per-RNA constraint pairs log ──────────────────────────────────────
        log_rows = (
            [{"pos_i": i, "pos_j": j, "count": c, "status": "KEPT"}
             for i, j, c in kept] +
            [{"pos_i": i, "pos_j": j, "count": c, "status": "REMOVED_multi_partner"}
             for i, j, c in raw
             if (min(i,j), max(i,j), c) not in [(a,b,cc) for a,b,cc in after_mp]] +
            [{"pos_i": i, "pos_j": j, "count": c, "status": "REMOVED_pseudoknot"}
             for i, j, c in removed_pk]
        )
        log_path = out_dir / f"{safe}_constraint_pairs.tsv"
        pd.DataFrame(log_rows).sort_values("pos_i").to_csv(log_path, sep="\t", index=False)

        print(f"  → Per-RNA FASTA: {fasta_path.name}")
        print(f"  → Pairs log:     {log_path.name}")

        combined_con.append((rna_name, rna_u, constraint))

    # ── Combined constrained input: all RNAs in one file, 3 lines per entry ───
    con_input_path = out_dir / "all_rnafold_constrained_input.txt"
    with open(con_input_path, "w") as f:
        for rna_name, rna_u, constraint in combined_con:
            f.write(f">{rna_name}\n{rna_u}\n{constraint}\n")
    print(f"\nCombined constrained input ({len(combined_con)} RNAs) → {con_input_path.name}")
    for rna_name, _, _ in combined_con:
        print(f"  {rna_name.split('_URS')[0]}")

    # ── Combined unconstrained input: all RNAs in one file, 2 lines per entry ─
    uncon_input_path = out_dir / "all_rnafold_unconstrained_input.txt"
    with open(uncon_input_path, "w") as f:
        for rna_name, rna_u in combined_uncon:
            f.write(f">{rna_name}\n{rna_u}\n")
    print(f"\nCombined unconstrained input ({len(combined_uncon)} RNAs) → {uncon_input_path.name}")

    # ── Run script ────────────────────────────────────────────────────────────
    sh_path = out_dir / "run_rnafold.sh"
    with open(sh_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Run RNAfold for all RNAs — one command per condition.\n")
        f.write("# Output files contain all RNAs concatenated.\n")
        f.write("# compare_rnafold.py parses them by header name.\n\n")
        f.write("# Condition A: unconstrained baseline (pure thermodynamics)\n")
        f.write(f"RNAfold --noPS \\\n"
                f"  < {uncon_input_path} \\\n"
                f"  > {out_dir}/all_rnafold_unconstrained_output.txt\n\n")
        f.write("# Condition B: DDD hard-constrained\n")
        f.write(f"RNAfold --constraint --noPS \\\n"
                f"  < {con_input_path} \\\n"
                f"  > {out_dir}/all_rnafold_constrained_output.txt\n\n")

    print(f"\nRun script → {sh_path.name}")
    print(f"\nTo run both conditions:\n  bash {sh_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()