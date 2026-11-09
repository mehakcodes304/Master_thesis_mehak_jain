#!/usr/bin/env python3
"""
RNAnue benchmark analysis — nucleotide-fraction overlap version.

Changes from binary-overlap version:
  1. Overlap criterion: a detected arm is considered to match a known stem arm
     only if at least --min_overlap_fraction (default 0.50) of the DETECTED
     arm's nucleotides fall inside the known stem arm's boundaries.
     The old version accepted any 1-nt overlap; this version requires ≥50 % of
     the detected span to be inside the stem.

  2. Both arms must independently satisfy the fraction threshold (forward or
     reverse orientation).  The nearest-midpoint rule still resolves ties.

  3. The --min_overlap_fraction flag (float, 0–1) controls the threshold so it
     can be swept without editing the script.

  4. [NEW] --simulated_reads: path to the FASTA/FASTQ produced by the chimeric
     read simulator.  Only stems whose reads appear in that file are included in
     the ground-truth denominator.  Stems that were silently filtered during
     simulation (arm too short, junction placement failure, etc.) are excluded
     from TP/FP/FN and from the recall denominator — RNAnue never received
     evidence for them, so they must not count as False Negatives.

     Read headers are expected to contain the stem name and interaction number
     in the form:
         >{rna_name}_stem_{interaction_number}_{anything}
     e.g. >rrsA_stem_3_fwd_0001

     If the header format differs, adjust HEADER_RE below (or supply
     --simulated_manifest instead; see below).

  4b.[NEW] --simulated_manifest: alternative to --simulated_reads.  A TSV or
     CSV with columns "name" and "interaction_number" listing exactly the stems
     that produced at least one read.  Useful if you already have a simulation
     log.  Mutually exclusive with --simulated_reads.

     If neither --simulated_reads nor --simulated_manifest is given, the script
     falls back to using ALL stems in the stems2 CSV (old behaviour) and prints
     a prominent warning.

All other logic (coordinate system, TP/FP/FN definitions, intermolecular
handling, precision/recall/F1) is unchanged from the previous version.

Coordinate system reminder:
  - RNAnue TSV (fst_seg_strt / sec_seg_strt): 0-based, inclusive end.
  - stems2 sub_interaction_stem: 1-based inclusive → converted to 0-based here.
"""

import argparse
import re
import sys
import pandas as pd


# =============================================================================
# HEADER REGEX — adjust if your simulator uses a different naming convention.
# Expected format: >{rna_name}_stem_{interaction_number}_{anything}
# Group 1 = rna_name, Group 2 = interaction_number
# =============================================================================

HEADER_RE = re.compile(r"^>(.+?)(?:_URS[A-Z0-9]+_\d+)?_int(\d+)_read")


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--interactions",
        default=(
            "/mnt/ceph/mjain/software/envs/bio/rnanue_run/ecoli_rRNA/output/04_analyze/treatment/condition1/condition1_interactions.tsv"
        ),
        help="Path to condition1_interactions.tsv (RNAnue output)",
    )
    p.add_argument(
        "--stems",
        default=(
            "/mnt/ceph/mjain/software/envs/bio/stems/ecoli_rRNA/stems2_ecoli_rRNA.csv"
        ),
        help="Path to stems2_*.csv (ground truth)",
    )
    p.add_argument(
        "--min_overlap_fraction",
        type=float,
        default=0.50,
        help=(
            "Minimum fraction of a DETECTED arm's nucleotides that must fall "
            "inside the known stem arm for the arm to count as matched. "
            "Default: 0.50 (50 %%)."
        ),
    )

    sim_group = p.add_mutually_exclusive_group()
    sim_group.add_argument(
        "--simulated_reads",
        metavar="FASTA",
        help=(
            "FASTA (or FASTQ) file produced by the chimeric read simulator. "
            "Read headers are parsed with the regex in HEADER_RE to identify "
            "which (name, interaction_number) pairs were actually simulated. "
            "Only those stems form the ground-truth denominator."
        ),
    )
    sim_group.add_argument(
        "--simulated_manifest",
        metavar="TSV_OR_CSV",
        help=(
            "Tab- or comma-separated file with columns 'name' and "
            "'interaction_number', listing stems that produced ≥1 read. "
            "Alternative to --simulated_reads when you already have a "
            "simulation log."
        ),
    )

    return p.parse_args()


# =============================================================================
# SIMULATED STEM SET
# =============================================================================

def load_simulated_stems_from_fasta(path: str) -> set:
    """
    Parse a FASTA or FASTQ file and extract (name, interaction_number) from
    every header line, using HEADER_RE.

    Returns the set of unique (name, int) tuples found.
    Warns about headers that did not match the regex.
    """
    simulated = set()
    unmatched = 0

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line.startswith(">") and not line.startswith("@"):
                continue
            # Strip leading > or @
            header = ">" + line[1:]
            m = HEADER_RE.match(header)
            if m:
                simulated.add((m.group(1), int(m.group(2))))
            else:
                unmatched += 1

    if unmatched:
        print(
            f"  WARNING: {unmatched} header lines did not match HEADER_RE and "
            f"were ignored.  Check that HEADER_RE matches your simulator's "
            f"header format.",
            file=sys.stderr,
        )
    return simulated


def load_simulated_stems_from_manifest(path: str) -> set:
    """
    Load (name, interaction_number) pairs from a TSV or CSV manifest.
    """
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    return set(zip(df["name"].astype(str), df["interaction_number"].astype(int)))


# =============================================================================
# COORDINATE HELPERS
# =============================================================================

def parse_core_arms_0based(sub_interaction_stem: str):
    """
    Parse sub_interaction_stem and return arm coords in 0-based inclusive format.

    stems2 stores 1-based inclusive coords, e.g. [(10-17), (2905-2912)].
    Converting to 0-based: subtract 1 from start only; end is inclusive in both.
    Returns ((arm1_start, arm1_end), (arm2_start, arm2_end)) sorted by start,
    or None on parse failure.
    """
    matches = re.findall(r"\((\d+)-(\d+)\)", str(sub_interaction_stem))
    if len(matches) < 2:
        return None
    arms = [(int(s) - 1, int(e) - 1) for s, e in matches[:2]]
    return tuple(sorted(arms, key=lambda a: a[0]))


def nucleotide_overlap_fraction(
    det_start: int,
    det_end:   int,
    ref_start: int,
    ref_end:   int,
) -> float:
    """
    Return the fraction of the REFERENCE (known stem arm) interval that is
    covered by the DETECTED (RNAnue cluster segment) interval.

    All coordinates are 0-based inclusive.

    We go nucleotide by nucleotide through the known arm and ask: how many of
    its nucleotides fall inside the detected cluster segment?

    Denominator = reference arm length.
    A known arm of 4 nt fully inside a 54-nt cluster span → 4/4 = 100%.
    A known arm of 4 nt half-outside the cluster span    → 2/4 =  50%.

    Formula:
        overlap_len = max(0, min(det_end, ref_end) - max(det_start, ref_start) + 1)
        fraction    = overlap_len / ref_length

    Returns 0.0 if the reference interval has zero length.
    """
    ref_len = ref_end - ref_start + 1
    if ref_len <= 0:
        return 0.0
    overlap_len = max(0, min(det_end, ref_end) - max(det_start, ref_start) + 1)
    return overlap_len / ref_len


def midpoint(start: int, end: int) -> float:
    return (start + end) / 2.0


# =============================================================================
# STEM MATCHING
# =============================================================================

def match_cluster_to_stems(
    fst_start: int,
    fst_end:   int,
    sec_start: int,
    sec_end:   int,
    candidates: pd.DataFrame,
    min_frac:   float,
) -> list:
    """
    Find ALL known stems matched by one RNAnue cluster.

    A stem is matched if:
      - Forward: ≥ min_frac of arm1's nucleotides fall in fst_seg,
                 AND ≥ min_frac of arm2's nucleotides fall in sec_seg.
      - Reverse: ≥ min_frac of arm2's nucleotides fall in fst_seg,
                 AND ≥ min_frac of arm1's nucleotides fall in sec_seg.

    One cluster can match multiple stems (e.g. a wide aggregate cluster that
    spans several adjacent stems on the same RNA).  All such stems are
    credited — RNAnue genuinely detected each of them.

    Parameters
    ----------
    fst_start, fst_end : 0-based inclusive — first segment of the detected cluster
    sec_start, sec_end : 0-based inclusive — second segment of the detected cluster
    candidates         : rows from stems2 for this RNA
    min_frac           : minimum fraction of each KNOWN ARM covered by the cluster
                         segment (denominator = reference arm length)

    Returns list of (name, interaction_number) tuples, possibly empty.
    """
    matches = []

    for _, row in candidates.iterrows():
        parsed = parse_core_arms_0based(row["sub_interaction_stem"])
        if parsed is None:
            continue
        arm1, arm2 = parsed  # arm1 has smaller start position

        # --- Forward orientation: fst segment covers arm1, sec segment covers arm2 ---
        fwd = (
            nucleotide_overlap_fraction(fst_start, fst_end, arm1[0], arm1[1]) >= min_frac
            and
            nucleotide_overlap_fraction(sec_start, sec_end, arm2[0], arm2[1]) >= min_frac
        )

        # --- Reverse orientation: fst segment covers arm2, sec segment covers arm1 ---
        rev = (
            nucleotide_overlap_fraction(fst_start, fst_end, arm2[0], arm2[1]) >= min_frac
            and
            nucleotide_overlap_fraction(sec_start, sec_end, arm1[0], arm1[1]) >= min_frac
        )

        if fwd or rev:
            matches.append((str(row["name"]), int(row["interaction_number"])))

    return matches


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    detected = pd.read_csv(args.interactions, sep="\t")
    known    = pd.read_csv(args.stems)

    # -------------------------------------------------------------------------
    # GROUND-TRUTH STEM SET
    # Only stems that were actually simulated count toward TP/FP/FN/recall.
    # Filtered-out stems (arm too short, junction placement failed, etc.) are
    # excluded — RNAnue received no reads for them, so they cannot be FNs.
    # -------------------------------------------------------------------------

    all_known_stems = set(zip(known["name"].astype(str),
                              known["interaction_number"].astype(int)))

    if args.simulated_reads:
        simulated_stems = load_simulated_stems_from_fasta(args.simulated_reads)
        print(f"Simulated reads file:  {args.simulated_reads}")
        print(f"  Stems found in FASTA: {len(simulated_stems)}")
        # Restrict ground truth to simulated stems only
        ground_truth_stems = all_known_stems & simulated_stems
        excluded = all_known_stems - simulated_stems
        if excluded:
            print(f"  Stems in stems2 but NOT simulated (excluded from benchmark): "
                  f"{len(excluded)}")
            for name, inum in sorted(excluded):
                print(f"    {name} int{inum}")
        print()

    elif args.simulated_manifest:
        simulated_stems = load_simulated_stems_from_manifest(args.simulated_manifest)
        print(f"Simulated manifest:    {args.simulated_manifest}")
        print(f"  Stems in manifest:    {len(simulated_stems)}")
        ground_truth_stems = all_known_stems & simulated_stems
        excluded = all_known_stems - simulated_stems
        if excluded:
            print(f"  Stems in stems2 but NOT simulated (excluded from benchmark): "
                  f"{len(excluded)}")
            for name, inum in sorted(excluded):
                print(f"    {name} int{inum}")
        print()

    else:
        print(
            "WARNING: Neither --simulated_reads nor --simulated_manifest was supplied.\n"
            "  Using ALL stems in the stems2 CSV as the ground truth denominator.\n"
            "  This OVER-COUNTS false negatives if any stems were silently filtered\n"
            "  during simulation (arm too short, junction placement failure, etc.).\n"
            "  Provide --simulated_reads <fasta> for accurate benchmarking.\n",
            file=sys.stderr,
        )
        ground_truth_stems = all_known_stems

    # Restrict `known` DataFrame to only ground-truth stems for matching loop
    known_gt = known[
        known.apply(
            lambda r: (str(r["name"]), int(r["interaction_number"])) in ground_truth_stems,
            axis=1,
        )
    ].copy()

    # Intramolecular = both arms on the same RNA feature
    detected["intramolecular"] = detected["fst_feat_id"] == detected["sec_feat_id"]
    intra = detected[detected["intramolecular"]].copy()
    inter = detected[~detected["intramolecular"]].copy()

    print(f"RNAnue predictions:")
    print(f"  Intramolecular:  {len(intra)}")
    print(f"  Intermolecular:  {len(inter)}  (out-of-scope — reported separately)")
    print(f"Ground-truth stems (simulated): {len(ground_truth_stems)}")
    print(f"  (stems2 total: {len(all_known_stems)})")
    print(f"Min overlap fraction: {args.min_overlap_fraction:.0%}  "
          f"(each detected arm must have ≥{args.min_overlap_fraction:.0%} of its "
          f"nucleotides inside the known stem arm)")
    print()

    # -------------------------------------------------------------------------
    # MATCHING LOOP
    #
    # One cluster can match MULTIPLE stems: a wide aggregate cluster that spans
    # several adjacent stems credits all of them.
    #
    # Stem-level TP:  a stem matched by ≥1 cluster.
    # Cluster-level TP: a cluster that matched ≥1 stem.
    # Cluster-level FP: a cluster that matched 0 stems.
    # -------------------------------------------------------------------------

    matched_stems:    set  = set()   # (name, interaction_number) — recalled stems
    tp_clusters:      int  = 0       # clusters that matched ≥1 stem
    fp_clusters:      list = []      # cluster IDs that matched 0 stems

    for _, row in intra.iterrows():
        rna_name  = str(row["fst_feat_id"]).replace("gene_", "")
        fst_start = int(row["fst_seg_strt"])
        fst_end   = int(row["fst_seg_end"])
        sec_start = int(row["sec_seg_strt"])
        sec_end   = int(row["sec_seg_end"])

        # Only consider ground-truth stems for this RNA
        candidates = known_gt[known_gt["name"] == rna_name]

        cluster_matches = []
        if not candidates.empty:
            cluster_matches = match_cluster_to_stems(
                fst_start, fst_end, sec_start, sec_end,
                candidates, args.min_overlap_fraction,
            )

        if cluster_matches:
            tp_clusters += 1
            matched_stems.update(cluster_matches)
        else:
            fp_clusters.append(row.get("cluster_ID", "unknown"))

    # Intermolecular clusters are always FP (only intramolecular reads simulated)
    inter_fp = list(inter.get("cluster_ID", inter.index.astype(str)))

    # -------------------------------------------------------------------------
    # COUNTS
    # -------------------------------------------------------------------------

    missed_stems = ground_truth_stems - matched_stems

    # Stem-level metrics
    n_tp_stems = len(matched_stems)
    n_fn_stems = len(missed_stems)

    # Cluster-level precision: TP clusters / all clusters (intra + inter)
    total_intra = len(intra)
    total_all   = len(intra) + len(inter)
    n_fp_clusters = len(fp_clusters) + len(inter)   # intra FP + all inter

    recall    = n_tp_stems / len(ground_truth_stems) if ground_truth_stems else 0.0
    precision = tp_clusters / total_all              if total_all          else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    print("=== Matched stems (True Positives — stem level) ===")
    for name, int_num in sorted(matched_stems):
        rows = known_gt[(known_gt["name"] == name) &
                        (known_gt["interaction_number"] == int_num)]
        if not rows.empty:
            print(f"  {name} int{int_num}: {rows.iloc[0]['sub_interaction_stem']}")

    print()
    print("=== Missed stems (False Negatives — stem level) ===")
    for name, int_num in sorted(missed_stems):
        rows = known_gt[(known_gt["name"] == name) &
                        (known_gt["interaction_number"] == int_num)]
        if not rows.empty:
            print(f"  {name} int{int_num}: {rows.iloc[0]['sub_interaction_stem']}")

    print()
    print("=== Unmatched intramolecular clusters (Intra FP — cluster level) ===")
    for cid in fp_clusters:
        print(f"  {cid}")

    print()
    print("=== Results ===")
    print(f"Known stems in stems2 CSV:              {len(all_known_stems)}")
    print(f"  of which simulated:                   {len(ground_truth_stems)}")
    print(f"  excluded (not simulated):             {len(all_known_stems) - len(ground_truth_stems)}")
    print(f"Intramolecular clusters (RNAnue):       {total_intra}")
    print(f"Intermolecular clusters (out-of-scope): {len(inter)}")
    print()
    print(f"--- Stem-level ---")
    print(f"True Positives  (TP stems):             {n_tp_stems}")
    print(f"False Negatives (FN stems):             {n_fn_stems}")
    print()
    print(f"--- Cluster-level ---")
    print(f"TP clusters (matched ≥1 stem):          {tp_clusters}")
    print(f"FP clusters (matched 0 stems):          {n_fp_clusters}")
    print(f"  of which intramolecular FP:           {len(fp_clusters)}")
    print(f"  of which intermolecular (always FP):  {len(inter)}")
    print()
    print(f"Recall:     {recall:.1%}   (TP stems / simulated stems)")
    print(f"Precision:  {precision:.1%}   (TP clusters / all clusters)")
    print(f"F1 score:   {f1:.1%}")


if __name__ == "__main__":
    main()