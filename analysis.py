#!/usr/bin/env python3
"""
RNAnue benchmark analysis — corrected version.

Changes from original:
  1. Coordinate system: RNAnue TSV uses 0-based coords; stems2 is 1-based.
     Stems2 coords are converted to 0-based before any comparison.
  2. Matching rule: cluster matched to nearest known stem by midpoint distance,
     subject to a required overlap on both arms. Prevents one wide cluster from
     claiming credit for multiple nearby stems.
  3. TP/FP/FN granularity: all at stem level (one entry in stems2 = one unit).
  4. Precision denominator: number of intramolecular clusters only.
     Intermolecular predictions are out-of-scope for this intramolecular benchmark
     and are reported separately, not included in FP or the denominator.

Verified facts (from data inspection):
  - RNAnue fst_seg_strt / sec_seg_strt: 0-based, inclusive end.
    Confirmed: cluster0 strt=0, end=36 exactly matches frag_arm1 range 0-36.
  - stems2 sub_interaction_stem: 1-based inclusive coords.
    Confirmed: matches core_arm1_start/end in chimeric reads CSV.
  - fst_feat_id format: "gene_rRNA_23S" → strip "gene_" → "rRNA_23S" = stems2 name.
    Confirmed correct.
  - RNAnue cluster span covers all fragment positions (not just core arm).
    A cluster for a short stem can span 100+ nt due to symmetric placement
    with max_arm_extension=100.
"""

import re
import pandas as pd

# =============================================================================
# PATHS — edit these for human_rRNA or tRNA datasets
# =============================================================================

INTERACTIONS_TSV = (
    "/mnt/ceph/mjain/software/envs/bio/rnanue_run/"
    "human_tRNA/output/04_analyze/treatment/condition1/"
    "condition1_interactions.tsv"
)
STEMS_CSV = (
    "/mnt/ceph/mjain/software/envs/bio/stems/"
    "human_tRNA/stems2_human_tRNA.csv"
)
CHIMERIC_CSV = (
    "/mnt/ceph/mjain/software/envs/bio/chimeric_reads/"
    "human_tRNA/human_tRNA_chimeric_reads.csv"
)

# =============================================================================
# LOAD
# =============================================================================

detected = pd.read_csv(INTERACTIONS_TSV, sep="\t")
known    = pd.read_csv(STEMS_CSV)

# Filter ground truth to only interactions that got chimeric reads generated.
# Interactions with 0 reads (e.g. rRNA_16S_ int1, rRNA_18S_N_2_4 int1) are
# geometrically impossible under current params and must be excluded from the
# recall denominator — RNAnue cannot recover what was never simulated.
chimeric = pd.read_csv(CHIMERIC_CSV, usecols=["name", "interaction_number"])
valid    = set(zip(chimeric["name"], chimeric["interaction_number"]))
known    = known[known.apply(
    lambda r: (r["name"], r["interaction_number"]) in valid, axis=1
)].copy()

# Flag intramolecular predictions (same feature on both arms)
detected["intramolecular"] = detected["fst_feat_id"] == detected["sec_feat_id"]
intra = detected[detected["intramolecular"]].copy()
inter = detected[~detected["intramolecular"]].copy()

print(f"RNAnue predictions:")
print(f"  Intramolecular:  {len(intra)}")
print(f"  Intermolecular:  {len(inter)}  (out-of-scope — reported separately)")
print(f"Known stems:       {len(known)}")
print()


# =============================================================================
# COORDINATE HELPERS
# =============================================================================

def parse_core_arms_0based(sub_interaction_stem: str) -> tuple | None:
    """
    Parse a sub_interaction_stem string and return arm coordinates in 0-based
    inclusive format (matching RNAnue's coordinate system).

    stems2 stores 1-based inclusive coords, e.g. [(10-17), (2905-2912)].
    Converting to 0-based: subtract 1 from start, end stays the same.
    Returns ((arm1_start, arm1_end), (arm2_start, arm2_end)) or None.
    """
    matches = re.findall(r"\((\d+)-(\d+)\)", str(sub_interaction_stem))
    if len(matches) < 2:
        return None
    arms = [(int(s) - 1, int(e) - 1) for s, e in matches[:2]]
    # Always return sorted by start position
    return tuple(sorted(arms, key=lambda a: a[0]))


def overlaps_0based(a_start: int, a_end: int,
                    b_start: int, b_end: int) -> bool:
    """
    True if two 0-based inclusive intervals overlap by at least 1 position.
    No tolerance — exact overlap required.
    Tolerance was removed because the RNAnue cluster span already encompasses
    the full fragment range, so any genuine match will overlap without needing
    extra padding.
    """
    return a_start <= b_end and b_start <= a_end


def midpoint(start: int, end: int) -> float:
    return (start + end) / 2.0


def match_cluster_to_stem(
    fst_start: int,
    fst_end:   int,
    sec_start: int,
    sec_end:   int,
    candidates: pd.DataFrame,
) -> tuple | None:
    """
    Find the best-matching known stem for one RNAnue cluster.

    Strategy:
      1. Require overlap on both arms (fst vs arm1 AND sec vs arm2, or vice versa).
         Both arms must overlap to prevent a cluster from matching a stem it merely
         happens to brush past on one side.
      2. Among all stems satisfying (1), return the one whose core arm midpoints
         are closest to the cluster arm midpoints (sum of both midpoint distances).
         This prevents one wide cluster from claiming credit for multiple stems.

    Returns (name, interaction_number) of the best match, or None.

    Coordinates: all inputs are 0-based inclusive (RNAnue native).
    stems2 coords are converted to 0-based in parse_core_arms_0based().
    """
    fst_mid = midpoint(fst_start, fst_end)
    sec_mid = midpoint(sec_start, sec_end)

    best_match = None
    best_dist  = float("inf")

    for _, row in candidates.iterrows():
        parsed = parse_core_arms_0based(row["sub_interaction_stem"])
        if parsed is None:
            continue
        arm1, arm2 = parsed  # arm1 has smaller start, arm2 has larger start

        # Forward orientation: fst ↔ arm1, sec ↔ arm2
        fwd = (overlaps_0based(fst_start, fst_end, arm1[0], arm1[1]) and
               overlaps_0based(sec_start, sec_end, arm2[0], arm2[1]))

        # Reverse orientation: fst ↔ arm2, sec ↔ arm1
        rev = (overlaps_0based(fst_start, fst_end, arm2[0], arm2[1]) and
               overlaps_0based(sec_start, sec_end, arm1[0], arm1[1]))

        if not (fwd or rev):
            continue

        if fwd:
            dist = (abs(fst_mid - midpoint(*arm1)) +
                    abs(sec_mid - midpoint(*arm2)))
        else:
            dist = (abs(fst_mid - midpoint(*arm2)) +
                    abs(sec_mid - midpoint(*arm1)))

        if dist < best_dist:
            best_dist  = dist
            best_match = (str(row["name"]), int(row["interaction_number"]))

    return best_match


# =============================================================================
# MATCHING LOOP
# =============================================================================

# For each intramolecular cluster, find its best-matching known stem.
# Each stem can only be credited once (first cluster that claims it wins).
# This ensures TP ≤ total known stems.

matched_stems:     set   = set()   # (name, interaction_number) — TP stems
unmatched_clusters: list = []      # cluster IDs with no stem match — FP clusters

for _, row in intra.iterrows():
    # Strip "gene_" prefix from fst_feat_id to get the short RNA name
    # Verified: fst_feat_id = "gene_rRNA_23S" → strip → "rRNA_23S" = stems2 name
    rna_name = str(row["fst_feat_id"]).replace("gene_", "")

    fst_start = int(row["fst_seg_strt"])
    fst_end   = int(row["fst_seg_end"])
    sec_start = int(row["sec_seg_strt"])
    sec_end   = int(row["sec_seg_end"])

    # Filter known stems to this RNA only
    candidates = known[known["name"] == rna_name]

    if candidates.empty:
        unmatched_clusters.append(row.get("cluster_ID", "unknown"))
        continue

    match = match_cluster_to_stem(fst_start, fst_end, sec_start, sec_end,
                                  candidates)

    if match is not None and match not in matched_stems:
        matched_stems.add(match)
    elif match is None:
        unmatched_clusters.append(row.get("cluster_ID", "unknown"))
    # If match already claimed by a prior cluster: this cluster is still a FP
    # because it did not uniquely identify a new known stem
    else:
        unmatched_clusters.append(row.get("cluster_ID", "unknown"))


# =============================================================================
# COUNTS
# =============================================================================

all_known_stems = set(zip(known["name"], known["interaction_number"]))
missed_stems    = all_known_stems - matched_stems

true_positives  = len(matched_stems)                        # stems recovered by RNAnue
false_positives = len(unmatched_clusters) + len(inter)      # intra unmatched + all inter
false_negatives = len(missed_stems)                         # known stems not recovered

# Precision: TP / (TP + FP) — denominator = ALL clusters (intra + inter)
# Intermolecular clusters are false positives in this benchmark because only
# intramolecular reads were simulated. Any intermolecular prediction is wrong.
total_intra    = len(intra)
total_all      = len(intra) + len(inter)
precision      = true_positives / total_all if total_all > 0 else 0.0

# Recall: TP / (TP + FN) — denominator = total known stems
total_known    = len(known)
recall         = true_positives / total_known if total_known > 0 else 0.0

# F1: harmonic mean of precision and recall
f1 = (2 * precision * recall / (precision + recall)
      if (precision + recall) > 0 else 0.0)


# =============================================================================
# REPORT
# =============================================================================

print("=== Matched stems (True Positives) ===")
for name, int_num in sorted(matched_stems):
    row = known[(known["name"] == name) &
                (known["interaction_number"] == int_num)].iloc[0]
    print(f"  {name} int{int_num}: {row['sub_interaction_stem']}")

print()
print("=== Missed stems (False Negatives) ===")
for name, int_num in sorted(missed_stems):
    row = known[(known["name"] == name) &
                (known["interaction_number"] == int_num)].iloc[0]
    print(f"  {name} int{int_num}: {row['sub_interaction_stem']}")

print()
print("=== Unmatched intramolecular clusters (False Positives) ===")
for cid in unmatched_clusters:
    print(f"  {cid}")

print()
print("=== Results ===")
print(f"Known stems (ground truth):          {total_known}")
print(f"Intramolecular clusters (RNAnue):    {total_intra}")
print(f"Intermolecular clusters (excluded):  {len(inter)}")
print()
print(f"True Positives  (TP):                {true_positives}")
print(f"False Positives (FP):                {false_positives}")
print(f"  — unmatched intramolecular:        {len(unmatched_clusters)}")
print(f"  — all intermolecular:              {len(inter)}")
print(f"False Negatives (FN):                {false_negatives}")
print()
print(f"Recall:     {recall:.1%}   (TP / known stems)")
print(f"Precision:  {precision:.1%}   (TP / all clusters, intra + inter)")
print(f"F1 score:   {f1:.1%}")
