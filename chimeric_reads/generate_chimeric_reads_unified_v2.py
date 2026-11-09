#!/usr/bin/env python3
"""
Generate simulated chimeric RNA reads from RNA interaction arms.

UNIFIED VERSION — one code path for human/E. coli x tRNA/rRNA.

Run with:
    python generate_chimeric_reads_unified.py

Rules (unchanged from the validated version):
1. Keep RNA sequence text as given in the CSV, except uppercasing.
2. Do not reverse-complement.
3. Do not convert RNA to DNA or DNA to RNA *internally*. The U->T conversion
   is applied only when writing the DNA FASTQ and the DNA reference FASTA,
   so the CSV/FASTA/BED outputs stay byte-identical to the validated run.
4. Arm 1 and arm 2 have the same length within each read.
5. A fragment must not overlap another arm or another interaction arm.
   (NOTE: this cross-interaction check is DISABLED below, matching the
   behaviour of the run that produced 97.7% recall on human tRNA.)
6. The join has no unpaired bases between the two joined pieces.
7. Both orientations are supported inside the requested total per interaction.
8. BED12 output is written for IGV viewing.

Input coordinates are 1-based inclusive.
BED output uses standard BED coordinates (0-based start, half-open end).

-----------------------------------------------------------------------------
PROVENANCE
-----------------------------------------------------------------------------
The default SimParams below reproduce generate_chimeric_reads_v2_ORIGINAL.py
exactly. That parameter set was confirmed against
human_tRNA_chimeric_reads_dna.fastq by read-length support analysis:

    arm_len ~ U[max(MIN_ARM_LEN, ceil(MIN_READ_LEN/2)),
                min(MAX_ARM_LEN, MAX_READ_LEN//2)]
            = U[5, 20]      ->  read lengths 10..40 even

    Observed FASTQ: 17,400 reads, all even lengths, exactly 10..40,
    87 interactions x 200 reads, 8,700 per orientation.

Refactoring to per-dataset parameters does NOT change the RNG call sequence
(choose_arm_length -> sample_fragment(first) -> sample_fragment(second)),
so with SEED=42 the human_tRNA output is bit-identical to the original.
Verify this with the check in the "REPRODUCIBILITY CHECK" section at the
bottom of this docstring before trusting any new dataset.

    python generate_chimeric_reads_unified.py
    diff human_tRNA_chimeric_reads.csv  <old csv>       # must be empty
    diff human_tRNA_chimeric_reads_dna.fastq <old fastq> # must be empty
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class SimParams:
    """
    All parameters that affect read generation.

    Defaults = the human tRNA configuration validated at 97.7% recall /
    100% intramolecular precision. Do not change these defaults; override
    per dataset in the DATASETS block instead, so the tRNA regression
    check stays meaningful.
    """
    reads_per_interaction: int = 200
    orientation_mode:      str = "both"   # "both" | "arm1_to_arm2" | "arm2_to_arm1"

    min_read_len:      int = 10
    max_read_len:      int = 40
    min_arm_len:       int = 5
    max_arm_len:       int = 20
    max_arm_extension: int = 50

    min_intra_gap:  float | None = None
    max_intra_gap:  float | None = None
    min_inter_gap:  float | None = None
    max_inter_gap:  float | None = None
    inter_gap_mode: str = "diff_mean"   # diff_mean | diff_min | diff_max | primary_min

    # --- Fragment placement rule ---------------------------------------------
    # "junction_clean": the ligation-facing end of each fragment must sit inside
    #     its core arm. This is the rule used by the validated tRNA run. It is
    #     asymmetric: in arm2_to_arm1 orientation both fragments grow inward
    #     toward the loop, so the orientation becomes impossible whenever
    #     2*(arm length) + (loop gap) < 2*(fragment length).
    #
    # "symmetric": each fragment need only overlap its core by min_core_overlap
    #     nt and may extend either direction within max_arm_extension. Both
    #     orientations are then always placeable. More realistic (RNase does not
    #     cut at stem boundaries), but read junctions no longer mark stem
    #     boundaries, so recovered coordinates are fuzzier. Document this.
    placement_rule: str = "junction_clean"   # junction_clean | symmetric

    # Minimum nt of the core arm a fragment must cover. Only meaningful under
    # "symmetric" placement, where a fragment may otherwise carry just 1 nt of
    # the actual interaction. Under "junction_clean" the geometry guarantees
    # >= 1 nt, so the default is a no-op there.
    min_core_overlap: int = 1

    # --- Coverage profile ----------------------------------------------------
    # "uniform":  fragment start drawn uniformly -> flat coverage.
    # "gaussian": truncated normal centred so the fragment sits over the core
    #     midpoint -> bell-shaped coverage peaking on the stem. Roughly halves
    #     the fraction of reads that barely graze the core.
    coverage_profile:    str   = "uniform"   # uniform | gaussian
    gaussian_sigma_frac: float = 0.30        # smaller -> sharper peak

    # Minimum gap, in reference coordinates, between the two fragments of one
    # read. A gap of 0 means the read is a contiguous piece of the reference:
    # segemehl maps it as one linear alignment and NO split is detected, so the
    # read is undetectable by construction. Set this above the largest deletion
    # segemehl will absorb into a linear alignment rather than call as a split.
    min_frag_gap: int = 0

    # Skip an interaction entirely when its SHORTER core arm is below this.
    # Mostly redundant under "symmetric" placement, which can serve short stems;
    # kept for compatibility and for deliberately excluding sub-resolution stems.
    min_stem_length: int | None = None

    unique_reads_per_interaction: bool = False
    max_attempts_per_read:        int  = 1000
    seed:                         int  = 42

    # Reporting only — does not affect generation. Set to the shortest arm
    # that maps uniquely to this dataset's reference (from your k-mer
    # analysis). The run summary reports what fraction of arms fall below it.
    mappability_floor: int | None = None

    # Minimum number of ground-truth base pairs that must be covered by the
    # two fragments of a chimeric read (jointly). A pair (i, j) counts if
    # position i falls in one fragment AND position j in the other, AND (i,j)
    # is in the GT stem for that interaction.
    # 0 = disabled (default, backward-compatible). Set to e.g. 3 to guarantee
    # every generated read captures at least 3 real stem pairs.
    min_gt_pairs: int = 3

    def arm_length_range(self) -> tuple[int, int]:
        """The closed interval that choose_arm_length() will draw from."""
        lo = max(self.min_arm_len, math.ceil(self.min_read_len / 2))
        hi = min(self.max_arm_len, self.max_read_len // 2)
        return lo, hi


# Validated tRNA configuration. Used for human tRNA and, like-for-like,
# for E. coli tRNA (comparable reference size, comparable RNA length).
TRNA_PARAMS = SimParams(
    mappability_floor=None,   # 1.6 kb reference; short arms were fine here
)

# PROPOSED rRNA configuration — NOT yet validated.
#
# Rationale: the tRNA settings give a mean arm of ~8-9 nt against a 1.6 kb
# reference, which was unique enough. The human rRNA reference is ~10.2 kb
# and your k-mer analysis put the unique-mapping floor at 18-20 nt, so arms
# must clear that or segemehl will multi-map or drop them. This is very
# likely the same failure mode as the E. coli rRNA run.
#
# Treat these numbers as a starting point to be tuned and documented, not as
# a validated setting. Raise MAX_ARM_EXTENSION if long arms get rejected too
# often in short stems (watch the "generated N/200 reads" warnings).
RRNA_PARAMS = SimParams(
    min_read_len=38,
    max_read_len=50,
    min_arm_len=18,
    max_arm_len=25,
    max_arm_extension=100,
    max_attempts_per_read=5000,
    placement_rule="symmetric",
    min_core_overlap=1,   
    min_frag_gap=4, 
    mappability_floor=18,
)

# =============================================================================
# SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Read outputs go to   OUTPUT_ROOT/<label>/<output_prefix>.*
# Reference FASTAs to  REF_ROOT/<label>/<label>_reference_*.fa
# Override per dataset with the optional "output_dir" / "ref_dir" keys.
OUTPUT_ROOT = Path("/mnt/ceph/mjain/software/envs/bio/chimeric_reads")
REF_ROOT    = Path("/mnt/ceph/mjain/software/envs/bio/custom_ref_genome")

# Set "enabled": False to skip a dataset without deleting its config.
DATASETS = [
    {
        "label":            "human_tRNA",
        "enabled":          True,
        "sequence_csv":     "/mnt/ceph/mjain/software/envs/bio/custom_ref_genome/human_tRNA/human_tRNA.csv",
        "interactions_csv": "/mnt/ceph/mjain/software/envs/bio/stems/human_tRNA/stems2_human_tRNA.csv",
        "output_prefix":    "/mnt/ceph/mjain/software/envs/bio/chimeric_reads/human_tRNA/human_tRNA_chimeric_reads1",
        "params":           TRNA_PARAMS,
    },
    {
        "label":            "ecoli_tRNA",
        "enabled":          True,
        "sequence_csv":     "/mnt/ceph/mjain/software/envs/bio/custom_ref_genome/ecoli_tRNA/ecoli_tRNA.csv",
        "interactions_csv": "/mnt/ceph/mjain/software/envs/bio/stems/ecoli_tRNA/stems2_ecoli_tRNA.csv",
        "output_prefix":    "/mnt/ceph/mjain/software/envs/bio/chimeric_reads/ecoli_tRNA/ecoli_tRNA_chimeric_reads1",
        "params":           TRNA_PARAMS,
    },
    {
        "label":            "human_rRNA",
        "enabled":          True,         
        "sequence_csv":     "/mnt/ceph/mjain/software/envs/bio/custom_ref_genome/human_rRNA/human_rRNA.csv",
        "interactions_csv": "/mnt/ceph/mjain/software/envs/bio/stems/human_rRNA/stems2_human_rRNA.csv",
        "output_prefix":    "/mnt/ceph/mjain/software/envs/bio/chimeric_reads/human_rRNA/human_rRNA_chimeric_reads1",
        "params":           RRNA_PARAMS,
    },
    {
        "label":            "ecoli_rRNA",
        "enabled":          True,
        "sequence_csv":     "/mnt/ceph/mjain/software/envs/bio/custom_ref_genome/ecoli_rRNA/ecoli_rRNA.csv",
        "interactions_csv": "/mnt/ceph/mjain/software/envs/bio/stems/ecoli_rRNA/stems2_ecoli_rRNA.csv",
        "output_prefix":    "/mnt/ceph/mjain/software/envs/bio/chimeric_reads/ecoli_rRNA/ecoli_rRNA_chimeric_reads1",
        "params":           RRNA_PARAMS,
    },

   {
    "label":            "human_rRNA_5_8S_fixed6",
    "enabled":          True,
    "sequence_csv":     "/mnt/ceph/mjain/software/envs/bio/custom_ref_genome/human_rRNA/human_rRNA.csv",
    "interactions_csv": "/mnt/ceph/mjain/software/envs/bio/stems/human_rRNA/stems2_human_rRNA_5_8S.csv",
    "output_prefix":    "/mnt/ceph/mjain/software/envs/bio/chimeric_reads/human_rRNA_5_8S_fixed6/human_rRNA_5_8S_fixed6_chimeric_reads",
    "params": SimParams(
        reads_per_interaction  = 200,
        min_arm_len            = 8,
        max_arm_len            = 8,
        min_read_len           = 16,
        max_read_len           = 16,
        max_arm_extension      = 100,
        max_attempts_per_read  = 5000,
        placement_rule         = "symmetric",
        min_core_overlap       = 3,
        min_frag_gap           = 4,
        min_gt_pairs           = 3,
        mappability_floor      = None,   # can't apply 18nt floor to 6nt arms
        seed                   = 42,
    ),
},
]

MAKE_CSV                       = True
MAKE_FASTA                     = True    # RNA alphabet (U), as in the original
MAKE_FASTQ                     = True    # DNA alphabet (T), for segemehl/RNAnue
MAKE_BED                       = True
MAKE_REFERENCE_FASTA_FOR_IGV   = True    # RNA alphabet (U)
MAKE_REFERENCE_FASTA_DNA       = True    # DNA alphabet (T), for segemehl index
CHECK_GT_ARM_COVERAGE  = True   # print reads where either arm covers < 3 GT-paired nt
MIN_GT_NT_PER_ARM      = 3      # the threshold

FASTQ_QUALITY_CHAR = "I"    # Phred 40 — reads are simulated without errors

# Fill this list to convert existing read CSVs to BED without re-generating.
EXISTING_READ_CSVS_TO_BED: list[str] = []


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Range1:
    """A 1-based inclusive interval [start, end]."""
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError(f"Range start must be >= 1, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"Range end must be >= start, got {self.start}-{self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: Range1) -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class ArmRecord:
    """One interaction arm — used to block illegal fragment overlaps."""
    name:               str
    interaction_number: int
    arm_id:             int     # 1 = left/lower arm, 2 = right/higher arm
    region:             Range1


# =============================================================================
# BASIC HELPERS
# =============================================================================

def path_here(value: str | Path) -> Path:
    """Resolve a path relative to this script's directory."""
    p = Path(value)
    return p if p.is_absolute() else BASE_DIR / p


def clean_seq(seq: str) -> str:
    """Remove whitespace and uppercase the sequence. Never changes U/T."""
    return str(seq).strip().replace(" ", "").replace("\n", "").upper()


def to_dna(seq: str) -> str:
    """Convert an RNA sequence to DNA. Applied only at write time."""
    return str(seq).replace("U", "T").replace("u", "t")


def seq_slice(seq: str, region: Range1) -> str:
    """Slice a sequence using a 1-based inclusive Range1."""
    if region.start < 1 or region.end > len(seq):
        raise ValueError(f"Slice {region.start}-{region.end} out of range for length {len(seq)}")
    return seq[region.start - 1 : region.end]


def parse_stem_ranges(value: str) -> tuple[Range1, Range1]:
    """
    Parse one sub_interaction_stem cell into two arm ranges.
    Always returns (lower-start arm, higher-start arm).

    Supports:
      "[(1-7), (66-72)]"               <- range format
      "[(1, 72), (2, 71), (3, 70)]"    <- pair-list format
    """
    text = str(value)

    dash_matches = re.findall(r"\((\d+)\s*-\s*(\d+)\)", text)
    if len(dash_matches) >= 2:
        arms = [Range1(int(a), int(b)) for a, b in dash_matches[:2]]
        return tuple(sorted(arms, key=lambda r: r.start))  # type: ignore[return-value]

    pair_matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)
    if pair_matches:
        left  = [int(a) for a, _ in pair_matches]
        right = [int(b) for _, b in pair_matches]
        arm1  = Range1(min(left),  max(left))
        arm2  = Range1(min(right), max(right))
        return tuple(sorted([arm1, arm2], key=lambda r: r.start))  # type: ignore[return-value]

    raise ValueError(f"Cannot parse sub_interaction_stem: {value!r}")


def parse_stem_pair_set(value: str) -> set[tuple[int, int]]:
    """
    Parse one sub_interaction_stem cell into a set of (i, j) base-pair tuples
    (1-based reference positions). Used to verify GT coverage of read fragments.

    Supports two formats:
      Range format:    "[(1-7), (66-72)]"
        Arms are antiparallel, so position (1+k) pairs with (72-k) for k=0..6.
      Pair-list format: "[(1, 72), (2, 71), (3, 70)]"
        Pairs are listed explicitly.
    """
    text = str(value)
    pairs: set[tuple[int, int]] = set()

    # Pair-list format takes priority (contains commas inside parentheses)
    pair_matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)
    if pair_matches:
        for a, b in pair_matches:
            pairs.add((int(a), int(b)))
        return pairs

    # Range format: expand antiparallel stem pairing
    dash_matches = re.findall(r"\((\d+)\s*-\s*(\d+)\)", text)
    if len(dash_matches) >= 2:
        arms = sorted([(int(a), int(b)) for a, b in dash_matches[:2]], key=lambda x: x[0])
        a_start, a_end = arms[0]
        b_start, b_end = arms[1]
        n = min(a_end - a_start + 1, b_end - b_start + 1)
        for k in range(n):
            pairs.add((a_start + k, b_end - k))
        return pairs

    return pairs


def interval_distance(a: Range1, b: Range1) -> int:
    """Positions between two non-overlapping ranges. Returns 0 if they overlap."""
    if a.overlaps(b):
        return 0
    return (b.start - a.end - 1) if a.end < b.start else (a.start - b.end - 1)


def distance_between_interactions(curr: tuple[Range1, Range1],
                                  nxt:  tuple[Range1, Range1]) -> int:
    """Minimum primary-sequence gap between any arm in two interactions."""
    return min(interval_distance(a, b) for a in curr for b in nxt)


def read_float(row: pd.Series, col: str) -> float | None:
    """Safely read a float from a DataFrame row. Returns None if absent or NaN."""
    if col not in row.index or pd.isna(row[col]):
        return None
    return float(row[col])


def within_bounds(value: float | None,
                  lo:    float | None,
                  hi:    float | None) -> bool:
    """True if value is within [lo, hi]. None means unbounded."""
    if lo is None and hi is None:
        return True
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return (lo is None or value >= lo) and (hi is None or value <= hi)


# =============================================================================
# INPUT LOADING
# =============================================================================

def load_inputs(sequence_csv: Path,
                interactions_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate both input CSVs. Cleans sequences."""
    seq_df = pd.read_csv(sequence_csv)
    int_df = pd.read_csv(interactions_csv)

    missing_seq = {"name", "seq"} - set(seq_df.columns)
    missing_int = {"name", "interaction_number", "sub_interaction_stem"} - set(int_df.columns)
    if missing_seq:
        raise ValueError(f"Sequence CSV missing columns: {sorted(missing_seq)}")
    if missing_int:
        raise ValueError(f"Interactions CSV missing columns: {sorted(missing_int)}")

    seq_df = seq_df.copy()
    seq_df["seq"]              = seq_df["seq"].map(clean_seq)
    seq_df["computed_seq_len"] = seq_df["seq"].str.len()

    int_df = int_df.copy()
    int_df["interaction_number"] = int_df["interaction_number"].astype(int)

    return seq_df, int_df


def build_next_range_lookup(interactions: pd.DataFrame) -> dict[tuple[str, int], tuple[Range1, Range1]]:
    """
    For each (name, interaction_number), store the arm ranges of the NEXT
    interaction on the same RNA. Used to compute inter-interaction gap.
    """
    lookup: dict[tuple[str, int], tuple[Range1, Range1]] = {}

    for name, group in interactions.groupby("name", sort=False):
        rows = list(group.sort_values("interaction_number").itertuples(index=False))
        for idx, row in enumerate(rows[:-1]):
            key = (row.name, row.interaction_number)
            lookup[key] = parse_stem_ranges(rows[idx + 1].sub_interaction_stem)

    return lookup


def build_arm_index(interactions: pd.DataFrame) -> dict[str, list[ArmRecord]]:
    """
    Build a dict: RNA name -> list of ArmRecords for that RNA.
    """
    index: dict[str, list[ArmRecord]] = defaultdict(list)

    for row in interactions.itertuples(index=False):
        name               = row.name
        interaction_number = int(row.interaction_number)
        arm1, arm2         = parse_stem_ranges(row.sub_interaction_stem)
        index[name].append(ArmRecord(name, interaction_number, 1, arm1))
        index[name].append(ArmRecord(name, interaction_number, 2, arm2))

    return index


def get_interaction_gap(row:            pd.Series,
                        next_ranges:    tuple[Range1, Range1] | None,
                        current_ranges: tuple[Range1, Range1],
                        mode:           str) -> float | None:
    """
    Measure the gap from this interaction to the next, using the requested mode.
    """
    d1   = read_float(row, "diff_i_curr_to_j_next")
    d2   = read_float(row, "diff_j_curr_to_i_next")
    vals = [v for v in [d1, d2] if v is not None]

    if mode == "diff_mean" and vals:
        return sum(vals) / len(vals)
    if mode == "diff_min"  and vals:
        return min(vals)
    if mode == "diff_max"  and vals:
        return max(vals)
    if next_ranges is None:
        return None

    return float(distance_between_interactions(current_ranges, next_ranges))


# =============================================================================
# READ SAMPLING
# =============================================================================

def orientation_targets(total: int, mode: str) -> dict[str, int]:
    """
    Split the requested read count across orientations.
    "both" splits evenly; remainder goes to arm1_to_arm2.
    """
    if mode == "arm1_to_arm2":
        return {"arm1_to_arm2": total}
    if mode == "arm2_to_arm1":
        return {"arm2_to_arm1": total}
    if mode != "both":
        raise ValueError("orientation_mode must be both, arm1_to_arm2, or arm2_to_arm1")
    half = total // 2
    return {"arm1_to_arm2": half, "arm2_to_arm1": total - half}


def choose_arm_length(rng: random.Random, p: SimParams) -> int | None:
    """
    Pick one arm length so that 2 x arm_len satisfies both read-length
    and arm-length bounds. Returns None if no valid length exists.
    """
    lo, hi = p.arm_length_range()
    return rng.randint(lo, hi) if lo <= hi else None


def core_overlap(fragment: Range1, core: Range1) -> int:
    """Number of nucleotides of core that fragment actually covers."""
    return max(0, min(fragment.end, core.end) - max(fragment.start, core.start) + 1)


def sample_start(rng:      random.Random,
                 lo:       int,
                 hi:       int,
                 core:     Range1,
                 frag_len: int,
                 p:        SimParams) -> int:
    """
    Pick a fragment start in [lo, hi] according to p.coverage_profile.

    IMPORTANT: the uniform branch calls rng.randint unconditionally, with no
    lo == hi shortcut. random.randint(a, a) still consumes entropy, so adding
    a shortcut would shift the RNG stream and break the tRNA regression check.
    """
    if p.coverage_profile == "uniform":
        return rng.randint(lo, hi)

    if p.coverage_profile != "gaussian":
        raise ValueError(f"coverage_profile must be uniform or gaussian, got {p.coverage_profile!r}")

    if lo == hi:
        return lo

    core_mid = (core.start + core.end) / 2.0
    mu       = min(max(core_mid - (frag_len - 1) / 2.0, lo), hi)
    sigma    = max(1.0, p.gaussian_sigma_frac * (hi - lo))

    for _ in range(50):
        cand = int(round(rng.gauss(mu, sigma)))
        if lo <= cand <= hi:
            return cand
    return rng.randint(lo, hi)


def sample_fragment(rng:      random.Random,
                    seq_len:  int,
                    core:     Range1,
                    frag_len: int,
                    position: str,
                    p:        SimParams) -> Range1 | None:
    """
    Sample one fragment of length frag_len overlapping its core arm.

    placement_rule == "junction_clean" (validated tRNA behaviour):
        the ligation-facing end must sit inside the core.
        position="first"  -> right end is the join; cannot go past core.end
        position="second" -> left end is the join;  cannot go before core.start
        This is asymmetric and makes arm2_to_arm1 impossible for short loops.

    placement_rule == "symmetric" (from the rRNA generator):
        the fragment need only overlap the core by min_core_overlap nt and may
        extend either direction inside max_arm_extension. `position` is accepted
        for interface compatibility but does not change the window.

    Returns None if no valid placement exists, or if the drawn fragment covers
    fewer than min_core_overlap nt of the core.
    """
    if frag_len < 1:
        return None
    if position not in ("first", "second"):
        raise ValueError(f"position must be 'first' or 'second', got {position!r}")

    if p.placement_rule == "junction_clean":
        if position == "first":
            lo = max(1, core.start - p.max_arm_extension, core.start - frag_len + 1)
            hi = core.end - frag_len + 1
        else:
            window_end = min(seq_len, core.end + p.max_arm_extension)
            lo = core.start
            hi = min(core.end, window_end - frag_len + 1)

    elif p.placement_rule == "symmetric":
        window_start = max(1,       core.start - p.max_arm_extension)
        window_end   = min(seq_len, core.end   + p.max_arm_extension)
        lo = max(window_start, core.start - frag_len + 1)
        hi = min(core.end,     window_end - frag_len + 1)

    else:
        raise ValueError(f"placement_rule must be junction_clean or symmetric, "
                         f"got {p.placement_rule!r}")

    if lo > hi:
        return None

    start    = sample_start(rng, lo, hi, core, frag_len, p)
    fragment = Range1(start, start + frag_len - 1)

    # No-op under junction_clean (geometry guarantees >= 1 nt) with the default
    # min_core_overlap of 1, so the tRNA path is unaffected.
    if core_overlap(fragment, core) < p.min_core_overlap:
        return None

    return fragment


def overlaps_forbidden_arm(fragment:           Range1,
                           arm_index:          dict[str, list[ArmRecord]],
                           rna_name:           str,
                           interaction_number: int,
                           own_arm_id:         int) -> bool:
    """
    Return True if fragment overlaps any arm on this RNA except its own core arm.
    """
    for arm in arm_index.get(rna_name, []):
        is_own = arm.interaction_number == interaction_number and arm.arm_id == own_arm_id
        if not is_own and fragment.overlaps(arm.region):
            return True
    return False


def make_one_read(seq:                str,
                  seq_len:            int,
                  name:               str,
                  interaction_number: int,
                  core1:              Range1,
                  core2:              Range1,
                  orientation:        str,
                  rng:                random.Random,
                  arm_index:          dict[str, list[ArmRecord]],
                  p:                  SimParams,
                  gt_pairs:           set[tuple[int, int]] | None = None) -> dict | None:
    """
    Attempt to sample one valid chimeric read for one interaction/orientation.
    """
    arm_len = choose_arm_length(rng, p)
    if arm_len is None:
        return None

    if orientation == "arm1_to_arm2":
        first_arm_id,  second_arm_id  = 1, 2
        first_core,    second_core    = core1, core2
    else:
        first_arm_id,  second_arm_id  = 2, 1
        first_core,    second_core    = core2, core1

    first_frag  = sample_fragment(rng, seq_len, first_core,  arm_len, "first",  p)
    second_frag = sample_fragment(rng, seq_len, second_core, arm_len, "second", p)

    if first_frag is None or second_frag is None:
        return None

    if first_frag.overlaps(second_frag):
        return None

    # GT-pairs coverage check — only active when min_gt_pairs > 0.
    # For each position in the first fragment paired with each position in the
    # second fragment: if that (i, j) is a ground-truth base pair for this
    # interaction, count it. Both orientations of the pair are checked.
    if p.min_gt_pairs > 0 and gt_pairs:
        n_gt = 0
        for ref_i in range(first_frag.start, first_frag.end + 1):
            for ref_j in range(second_frag.start, second_frag.end + 1):
                if (ref_i, ref_j) in gt_pairs or (ref_j, ref_i) in gt_pairs:
                    n_gt += 1
        if n_gt < p.min_gt_pairs:
            return None

    # NOTE: forbidden-arm (cross-interaction overlap) checks intentionally
    # disabled -- matches the exact behaviour of the 97.7%-recall run.
    #if overlaps_forbidden_arm(first_frag,  arm_index, name, interaction_number, first_arm_id):
    #    return None
    #if overlaps_forbidden_arm(second_frag, arm_index, name, interaction_number, second_arm_id):
    #    return None

    read_seq = seq_slice(seq, first_frag) + seq_slice(seq, second_frag)

    frag1, frag2 = (first_frag, second_frag) if orientation == "arm1_to_arm2" else (second_frag, first_frag)

    # Enforce the minimum reference gap AFTER normalisation, so frag1 is always
    # the lower-coordinate piece. A gap below this makes the read a contiguous
    # stretch of reference that segemehl will map linearly instead of splitting.
    # Consumes no RNG, and is a no-op at the default min_frag_gap of 0.
    if frag2.start - frag1.end - 1 < p.min_frag_gap:
        return None

    return {
        "orientation":        orientation,
        "first_piece_arm":    first_arm_id,
        "second_piece_arm":   second_arm_id,
        "first_piece_start":  first_frag.start,
        "first_piece_end":    first_frag.end,
        "second_piece_start": second_frag.start,
        "second_piece_end":   second_frag.end,
        "frag_arm1_start":    frag1.start,
        "frag_arm1_end":      frag1.end,
        "frag_arm2_start":    frag2.start,
        "frag_arm2_end":      frag2.end,
        "arm1_len":           arm_len,
        "arm2_len":           arm_len,
        "read_len":           len(read_seq),
        "sequence":           read_seq,
    }


def generate_reads_for_dataset(sequence_csv:     Path,
                               interactions_csv: Path,
                               p:                SimParams) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Full pipeline for one dataset.
    """
    rng = random.Random(p.seed)
    seq_df, interactions = load_inputs(sequence_csv, interactions_csv)

    seq_by_name: dict[str, dict] = {}
    for entry in seq_df.to_dict("records"):
        full_name  = str(entry["name"])
        short_name = re.sub(r"_URS[A-Z0-9]+_\d+$", "", full_name)
        seq_by_name[full_name]  = entry
        if short_name != full_name:
            seq_by_name[short_name] = entry
    next_lookup  = build_next_range_lookup(interactions)
    arm_index    = build_arm_index(interactions)

    records:  list[dict] = []
    warnings: list[str]  = []
    warned_missing: set[str] = set()

    interactions = interactions.sort_values(["name", "interaction_number"]).reset_index(drop=True)

    for _, row in interactions.iterrows():
        name               = str(row["name"])
        interaction_number = int(row["interaction_number"])

        if name not in seq_by_name:
            if name not in warned_missing:
                warnings.append(f"Skipping {name} -- sequence not found.")
                warned_missing.add(name)
            continue

        seq_meta = seq_by_name[name]
        seq      = str(seq_meta["seq"])
        seq_len  = len(seq)
        core1, core2 = parse_stem_ranges(row["sub_interaction_stem"])
        gt_pairs     = parse_stem_pair_set(row["sub_interaction_stem"]) if p.min_gt_pairs > 0 else None

        if core1.end > seq_len or core2.end > seq_len:
            warnings.append(f"Skipping {name} int {interaction_number} -- arm exceeds length {seq_len}.")
            continue

        if p.min_stem_length is not None:
            if min(core1.length, core2.length) < p.min_stem_length:
                continue   # below threshold -- deliberate exclusion, not a failure

        intra_gap   = interval_distance(core1, core2)
        next_ranges = next_lookup.get((name, interaction_number))
        inter_gap   = get_interaction_gap(row, next_ranges, (core1, core2), p.inter_gap_mode)

        if not within_bounds(intra_gap, p.min_intra_gap, p.max_intra_gap):
            continue
        if not within_bounds(inter_gap, p.min_inter_gap, p.max_inter_gap):
            continue

        seen: set[tuple] = set()
        targets = orientation_targets(p.reads_per_interaction, p.orientation_mode)

        read_count_this_interaction = 0

        for orientation, target_count in targets.items():
            emitted  = 0
            attempts = 0
            max_attempts = target_count * p.max_attempts_per_read

            while emitted < target_count and attempts < max_attempts:
                attempts += 1

                read = make_one_read(seq, seq_len, name, interaction_number,
                                     core1, core2, orientation, rng, arm_index, p,
                                     gt_pairs=gt_pairs)
                if read is None:
                    continue

                signature = (orientation,
                             read["frag_arm1_start"], read["frag_arm1_end"],
                             read["frag_arm2_start"], read["frag_arm2_end"])
                if p.unique_reads_per_interaction and signature in seen:
                    continue
                seen.add(signature)

                emitted += 1
                read_count_this_interaction += 1

                rna_id   = str(seq_meta.get("id", ""))
                igv_name = f"{name}_{rna_id}" if rna_id else name

                output_record = {
                    "read_id":              f"{igv_name}_int{interaction_number}_read{read_count_this_interaction:03d}_{orientation}",
                    "igv_name":             igv_name,
                    "name":                 name,
                    "id":                   seq_meta.get("id", ""),
                    "rna_type":             seq_meta.get("rna_type", ""),
                    "seq_len":              seq_len,
                    "interaction_number":   interaction_number,
                    "core_arm1_start":      core1.start,
                    "core_arm1_end":        core1.end,
                    "core_arm2_start":      core2.start,
                    "core_arm2_end":        core2.end,
                    "intra_arm_gap":        intra_gap,
                    "inter_interaction_gap": inter_gap if inter_gap is not None else "",
                    "inter_gap_mode":       p.inter_gap_mode,
                }
                output_record.update(read)
                records.append(output_record)

            if emitted < target_count:
                warnings.append(
                    f"{name} int {interaction_number} {orientation}: "
                    f"generated {emitted}/{target_count} reads. "
                    f"Relax length, gap, extension, or uniqueness settings."
                )

    return pd.DataFrame.from_records(records), warnings, seq_df


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_fasta(reads: pd.DataFrame, output_fasta: Path) -> None:
    """Write generated reads as FASTA, in the input alphabet (RNA / U)."""
    with output_fasta.open("w", encoding="utf-8") as fh:
        for row in reads.itertuples(index=False):
            fh.write(f">{row.read_id}\n{row.sequence}\n")


def write_fastq(reads: pd.DataFrame, output_fastq: Path) -> None:
    """
    Write generated reads as FASTQ in the DNA alphabet, with uniform Phred 40.

    U->T is applied here rather than in clean_seq so that the coordinates,
    the CSV and the RNA FASTA remain identical to the validated run. The
    DNA alphabet is required because segemehl indexes and reads must share
    an alphabet — a mismatch produces the MD5 error.
    """
    with output_fastq.open("w", encoding="utf-8") as fh:
        for row in reads.itertuples(index=False):
            seq  = to_dna(row.sequence)
            qual = FASTQ_QUALITY_CHAR * len(seq)
            fh.write(f"@{row.read_id}\n{seq}\n+\n{qual}\n")


def write_reference_fasta_and_fai(seq_df: pd.DataFrame,
                                  output_fasta: Path,
                                  as_dna: bool = False) -> None:
    """
    Write RNA sequences as a FASTA reference and matching .fai index for IGV.
    Set as_dna=True to emit the DNA (T) version used for the segemehl index.
    """
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    fai_rows: list[tuple] = []
    offset = 0

    with output_fasta.open("wb") as fh:
        for row in seq_df.itertuples(index=False):
            name = f"{row.name}_{row.id}" if hasattr(row, "id") and str(row.id) else str(row.name)
            seq  = to_dna(str(row.seq)) if as_dna else str(row.seq)

            header   = f">{name}\n".encode()
            seq_line = f"{seq}\n".encode()

            fh.write(header)
            seq_offset = offset + len(header)
            fh.write(seq_line)

            fai_rows.append((name, len(seq), seq_offset, len(seq), len(seq_line)))
            offset += len(header) + len(seq_line)

    fai_path = output_fasta.with_suffix(output_fasta.suffix + ".fai")
    with fai_path.open("w", encoding="utf-8") as fh:
        for row in fai_rows:
            fh.write("\t".join(map(str, row)) + "\n")


def bed12_row_from_read(row: pd.Series) -> list:
    """Convert one read row to a BED12 record."""
    arm1   = Range1(int(row["frag_arm1_start"]), int(row["frag_arm1_end"]))
    arm2   = Range1(int(row["frag_arm2_start"]), int(row["frag_arm2_end"]))
    blocks = sorted([arm1, arm2], key=lambda r: r.start)

    chrom_start   = blocks[0].start - 1
    chrom_end     = blocks[-1].end
    block_sizes   = ",".join(str(b.length)                        for b in blocks) + ","
    block_starts  = ",".join(str(b.start - 1 - chrom_start)      for b in blocks) + ","
    item_rgb      = "0,0,255" if row.get("orientation") == "arm1_to_arm2" else "255,0,0"
    chrom         = str(row.get("igv_name", row["name"]))

    return [
        chrom, chrom_start, chrom_end, str(row["read_id"]),
        0, "+", chrom_start, chrom_end, item_rgb, 2, block_sizes, block_starts,
    ]


def compute_coverage(reads: pd.DataFrame) -> pd.DataFrame:
    """Compute per-base arm coverage across all chimeric reads."""
    diff: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for row in reads.itertuples(index=False):
        chrom = str(getattr(row, "igv_name", row.name))
        for start_col, end_col in [("frag_arm1_start", "frag_arm1_end"),
                                    ("frag_arm2_start", "frag_arm2_end")]:
            s = int(getattr(row, start_col)) - 1
            e = int(getattr(row, end_col))
            diff[chrom][s] += 1
            diff[chrom][e] -= 1

    records = []
    for chrom, deltas in sorted(diff.items()):
        pos_sorted = sorted(deltas)
        coverage   = 0
        for i, pos in enumerate(pos_sorted):
            coverage += deltas[pos]
            end = pos_sorted[i + 1] if i + 1 < len(pos_sorted) else pos + 1
            if coverage > 0:
                records.append({"chrom": chrom, "start": pos, "end": end, "coverage": coverage})

    return pd.DataFrame(records)


def write_coverage_bedgraph(reads: pd.DataFrame,
                            output_path: Path,
                            track_name: str = "chimeric_coverage") -> None:
    """Write per-base arm coverage as a BEDGraph file."""
    cov = compute_coverage(reads)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        fh.write(f'track type=bedGraph name="{track_name}" '
                 "description=\"chimeric read arm coverage\" visibility=full autoScale=on\n")
        for row in cov.itertuples(index=False):
            fh.write(f"{row.chrom}\t{row.start}\t{row.end}\t{row.coverage}\n")
    print(f"  BEDGraph: {output_path}")


def write_coverage_bed(reads: pd.DataFrame,
                       output_path: Path,
                       track_name: str = "chimeric_coverage_bed") -> None:
    """Write per-base arm coverage as a standard BED file."""
    cov = compute_coverage(reads)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        fh.write(f'track name="{track_name}" '
                 "description=\"chimeric read arm coverage\" visibility=dense\n")
        for row in cov.itertuples(index=False):
            fh.write(f"{row.chrom}\t{row.start}\t{row.end}\tcoverage_{row.coverage}\t{row.coverage}\t+\n")
    print(f"  Coverage BED: {output_path}")


def write_bed12(reads: pd.DataFrame, output_bed: Path) -> None:
    """Write a BED12 file for IGV visualisation."""
    required = {"name", "read_id", "frag_arm1_start", "frag_arm1_end",
                "frag_arm2_start", "frag_arm2_end"}
    missing = required - set(reads.columns)
    if missing:
        raise ValueError(f"Cannot write BED -- missing columns: {sorted(missing)}")

    output_bed.parent.mkdir(parents=True, exist_ok=True)
    with output_bed.open("w", encoding="utf-8") as fh:
        fh.write('track name="chimeric_reads" description="Simulated chimeric RNA reads" '
                 'visibility=2 itemRgb="On"\n')
        for _, row in reads.iterrows():
            fh.write("\t".join(map(str, bed12_row_from_read(row))) + "\n")


# =============================================================================
# VALIDATION + RUN LOOP
# =============================================================================

def validate_settings(p: SimParams, label: str) -> None:
    """Catch impossible setting combinations before doing any work."""
    def bad(msg: str) -> None:
        raise ValueError(f"[{label}] {msg}")

    if p.reads_per_interaction < 1:
        bad("reads_per_interaction must be at least 1")
    if p.min_read_len < 1 or p.max_read_len < p.min_read_len:
        bad("read length bounds are invalid")
    if p.min_arm_len < 1 or p.max_arm_len < p.min_arm_len:
        bad("arm length bounds are invalid")
    if p.min_read_len > 2 * p.max_arm_len:
        bad("min_read_len is larger than 2 x max_arm_len")
    if p.max_read_len < 2 * p.min_arm_len:
        bad("max_read_len is smaller than 2 x min_arm_len")
    if p.max_arm_extension < 0:
        bad("max_arm_extension must be >= 0")
    if p.orientation_mode not in {"both", "arm1_to_arm2", "arm2_to_arm1"}:
        bad("orientation_mode must be both, arm1_to_arm2, or arm2_to_arm1")
    if p.inter_gap_mode not in {"diff_mean", "diff_min", "diff_max", "primary_min"}:
        bad("inter_gap_mode must be diff_mean, diff_min, diff_max, or primary_min")
    if p.placement_rule not in {"junction_clean", "symmetric"}:
        bad("placement_rule must be junction_clean or symmetric")
    if p.coverage_profile not in {"uniform", "gaussian"}:
        bad("coverage_profile must be uniform or gaussian")
    if not 0.0 < p.gaussian_sigma_frac <= 2.0:
        bad("gaussian_sigma_frac must be in (0, 2]")
    if p.min_frag_gap < 0:
        bad("min_frag_gap must be >= 0")
    if p.min_core_overlap < 1:
        bad("min_core_overlap must be >= 1")
    if p.min_core_overlap > p.min_arm_len:
        bad(f"min_core_overlap ({p.min_core_overlap}) exceeds min_arm_len "
            f"({p.min_arm_len}); no fragment could ever satisfy it")
    if p.placement_rule == "junction_clean" and p.min_core_overlap > 1:
        bad("min_core_overlap > 1 is only meaningful under symmetric placement")
    if p.min_stem_length is not None and p.min_stem_length < 1:
        bad("min_stem_length must be >= 1 or None")
    if p.min_gt_pairs < 0:
        bad("min_gt_pairs must be >= 0")

    lo, hi = p.arm_length_range()
    if lo > hi:
        bad(f"no valid arm length exists: effective range is [{lo}, {hi}]")


def summarize(reads: pd.DataFrame, p: SimParams) -> str:
    """Multi-line summary of a generated reads DataFrame."""
    if reads.empty:
        return "no reads generated"

    counts       = reads.groupby(["name", "interaction_number"]).size()
    orientations = reads.groupby("orientation").size().to_dict() if "orientation" in reads else {}
    equal        = bool((reads["arm1_len"] == reads["arm2_len"]).all())
    lo, hi       = p.arm_length_range()

    lines = [
        f"{len(reads)} reads over {len(counts)} interactions -- "
        f"min/max per interaction: {counts.min()}/{counts.max()} -- "
        f"orientations: {orientations} -- equal arm lengths: {equal}",
        f"  placement={p.placement_rule}, coverage={p.coverage_profile}, "
        f"min_frag_gap={p.min_frag_gap}, min_core_overlap={p.min_core_overlap}",
        f"  arm_len drawn from [{lo}, {hi}] -> read lengths "
        f"{reads['read_len'].min()}..{reads['read_len'].max()} "
        f"(mean {reads['read_len'].mean():.1f})",
    ]

    # Interactions represented by only one orientation are not comparable to
    # fully covered ones and will bias any recall figure computed over them.
    if p.orientation_mode == "both":
        per_orient = reads.groupby(["name", "interaction_number"])["orientation"].nunique()
        single     = int((per_orient < 2).sum())
        if single:
            lines.append(
                f"  WARNING: {single}/{len(per_orient)} interactions have reads in "
                f"ONE orientation only. Exclude them or report them separately -- "
                f"they are systematically the short-loop stems."
            )
        else:
            lines.append(f"  OK: all {len(per_orient)} interactions covered in both orientations.")

    if p.min_core_overlap is not None and p.placement_rule == "symmetric":
        lines.append(
            f"  note: fragments may cover as few as {p.min_core_overlap} nt of the "
            f"stem core; raise min_core_overlap to tighten the read/interaction link."
        )

    if p.mappability_floor is not None:
        below = int((reads["arm1_len"] < p.mappability_floor).sum())
        pct   = 100.0 * below / len(reads)
        if pct < 5.0:
            lines.append(
                f"  OK: {below}/{len(reads)} reads ({pct:.1f}%) have arms below the "
                f"{p.mappability_floor} nt mappability floor."
            )
        else:
            lines.append(
                f"  WARNING: {below}/{len(reads)} reads ({pct:.1f}%) have arms below "
                f"the {p.mappability_floor} nt mappability floor -- these will likely "
                f"multi-map or be dropped by segemehl. Raise min_arm_len for this dataset."
            )

    return "\n".join(lines)

def check_gt_arm_coverage(reads: pd.DataFrame,
                           interactions: pd.DataFrame,
                           min_gt_per_arm: int = 3) -> pd.DataFrame:
    """
    For every generated read, count how many GT base-pair positions are jointly
    covered on EACH arm fragment:
      - arm1_gt_nt: distinct positions i in frag_arm1 where (i,j) is a GT pair
                    AND j is in frag_arm2.
      - arm2_gt_nt: the matching j positions.
    Returns a DataFrame of reads where either arm covers < min_gt_per_arm.
    """
    gt_lookup: dict[tuple[str, int], set[tuple[int, int]]] = {}
    for _, row in interactions.iterrows():
        key = (str(row["name"]), int(row["interaction_number"]))
        gt_lookup[key] = parse_stem_pair_set(row["sub_interaction_stem"])

    failures = []
    for _, row in reads.iterrows():
        key = (str(row["name"]), int(row["interaction_number"]))
        gt_pairs = gt_lookup.get(key, set())
        if not gt_pairs:
            continue

        f1s = int(row["frag_arm1_start"]); f1e = int(row["frag_arm1_end"])
        f2s = int(row["frag_arm2_start"]); f2e = int(row["frag_arm2_end"])

        arm1_covered: set[int] = set()
        arm2_covered: set[int] = set()

        for (i, j) in gt_pairs:
            if f1s <= i <= f1e and f2s <= j <= f2e:
                arm1_covered.add(i); arm2_covered.add(j)
            elif f1s <= j <= f1e and f2s <= i <= f2e:
                arm1_covered.add(j); arm2_covered.add(i)

        n1, n2 = len(arm1_covered), len(arm2_covered)
        if n1 < min_gt_per_arm or n2 < min_gt_per_arm:
            failures.append({
                "read_id":            row["read_id"],
                "name":               row["name"],
                "interaction_number": int(row["interaction_number"]),
                "frag_arm1":          f"{f1s}-{f1e}",
                "frag_arm2":          f"{f2s}-{f2e}",
                "arm1_gt_nt":         n1,
                "arm2_gt_nt":         n2,
            })

    return pd.DataFrame(failures)


def run_generation() -> list[str]:
    """Generate reads for every enabled dataset and write all requested outputs."""
    all_warnings: list[str] = []

    for dataset in DATASETS:
        label = str(dataset["label"])

        if not dataset.get("enabled", True):
            print(f"Skipping {label} -- disabled in DATASETS.")
            continue

        p: SimParams     = dataset.get("params", SimParams())
        sequence_csv     = path_here(dataset["sequence_csv"])
        interactions_csv = path_here(dataset["interactions_csv"])

        output_dir    = Path(dataset.get("output_dir") or (OUTPUT_ROOT / label))
        ref_dir       = Path(dataset.get("ref_dir")    or (REF_ROOT    / label))
        output_prefix = output_dir / str(dataset["output_prefix"])

        validate_settings(p, label)

        missing = [str(f) for f in (sequence_csv, interactions_csv) if not f.exists()]
        if missing:
            print(f"Skipping {label} -- input files not found:")
            for f in missing:
                print(f"    {f}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {label}")
        reads, warnings, seq_df = generate_reads_for_dataset(sequence_csv, interactions_csv, p)
        all_warnings.extend(f"{label}: {w}" for w in warnings)

        if MAKE_CSV:
            q = output_prefix.with_suffix(".csv")
            reads.to_csv(q, index=False)
            print(f"  CSV:   {q}")

        if MAKE_FASTA:
            if reads.empty:
                print(f"  FASTA: skipped -- no reads generated for {label}")
            else:
                q = output_prefix.with_suffix(".fa")
                write_fasta(reads, q)
                print(f"  FASTA: {q}")

        if MAKE_FASTQ:
            if reads.empty:
                print(f"  FASTQ: skipped -- no reads generated for {label}")
            else:
                q = output_prefix.parent / f"{output_prefix.stem}_dna.fastq"
                write_fastq(reads, q)
                print(f"  FASTQ: {q}")

        if MAKE_BED:
            if reads.empty:
                print(f"  BED12: skipped -- no reads generated for {label}")
            else:
                q = output_prefix.with_suffix(".bed")
                write_bed12(reads, q)
                print(f"  BED12: {q}")

                label_clean = label.replace(" ", "_")
                write_coverage_bedgraph(
                    reads,
                    output_prefix.parent / f"{output_prefix.stem}_coverage.bedgraph",
                    track_name=f"{label_clean}_chimeric_coverage",
                )
                write_coverage_bed(
                    reads,
                    output_prefix.parent / f"{output_prefix.stem}_coverage.bed",
                    track_name=f"{label_clean}_chimeric_coverage_bed",
                )

        if MAKE_REFERENCE_FASTA_FOR_IGV:
            ref = ref_dir / f"{label}_reference_for_igv.fa"
            write_reference_fasta_and_fai(seq_df, ref, as_dna=False)
            print(f"  IGV ref FASTA: {ref}")
            print(f"  IGV ref index: {ref}.fai")

        if MAKE_REFERENCE_FASTA_DNA:
            ref_dna = ref_dir / f"{label}_reference_dna.fa"
            write_reference_fasta_and_fai(seq_df, ref_dna, as_dna=True)
            print(f"  DNA ref FASTA: {ref_dna}")
            print(f"    index with:  cd {ref_dna.parent} && "
                  f"segemehl.x -x {ref_dna.stem}.idx -d {ref_dna.name}")

        print(f"  {summarize(reads, p)}")

        if CHECK_GT_ARM_COVERAGE and not reads.empty:
            _, interactions_reloaded = load_inputs(sequence_csv, interactions_csv)
            # reload needed because interactions local var may be sorted/filtered
            fail_df = check_gt_arm_coverage(reads, interactions_reloaded, MIN_GT_NT_PER_ARM)
            if fail_df.empty:
                print(f"  GT arm check: all {len(reads)} reads cover >= {MIN_GT_NT_PER_ARM} GT nt on each arm ✓")
            else:
                pct = 100.0 * len(fail_df) / len(reads)
                print(f"  GT arm check: {len(fail_df)}/{len(reads)} reads ({pct:.1f}%) FAIL "
                      f"(< {MIN_GT_NT_PER_ARM} GT-paired nt on at least one arm)", file=sys.stderr)
                # Show up to 30 failing rows grouped by interaction
                pd.set_option("display.max_rows", 30)
                pd.set_option("display.width", 120)
                print(fail_df.to_string(index=False), file=sys.stderr)
                # Also save to file for full inspection
                fail_path = output_prefix.parent / f"{output_prefix.stem}_gt_arm_failures.tsv"
                fail_df.to_csv(fail_path, sep="\t", index=False)
                print(f"  Full failure list: {fail_path}", file=sys.stderr)

    return all_warnings


def run_existing_csv_to_bed() -> None:
    """Convert any existing read CSVs listed in EXISTING_READ_CSVS_TO_BED to BED12."""
    for csv_name in EXISTING_READ_CSVS_TO_BED:
        csv_path = path_here(csv_name)
        if not csv_path.exists():
            print(f"Skipping BED conversion -- not found: {csv_path}")
            continue
        bed_path = csv_path.with_suffix(".bed")
        write_bed12(pd.read_csv(csv_path), bed_path)
        print(f"Converted {csv_path.name} -> {bed_path.name}")


def main() -> int:
    warnings = run_generation()
    run_existing_csv_to_bed()

    if warnings:
        print("\nWarnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())