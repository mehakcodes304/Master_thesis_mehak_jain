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

    unique_reads_per_interaction: bool = False
    max_attempts_per_read:        int  = 1000
    seed:                         int  = 42

    # Reporting only — does not affect generation. Set to the shortest arm
    # that maps uniquely to this dataset's reference (from your k-mer
    # analysis). The run summary reports what fraction of arms fall below it.
    mappability_floor: int | None = None

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
    min_read_len=36,
    max_read_len=50,
    min_arm_len=18,
    max_arm_len=25,
    max_arm_extension=50,
    mappability_floor=18,
)


# =============================================================================
# SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Set "enabled": False to skip a dataset without deleting its config.
DATASETS = [
    {
        "label":            "human_tRNA",
        "enabled":          True,
        "sequence_csv":     "human_tRNA.csv",
        "interactions_csv": "stems2_human_tRNA.csv",
        "output_prefix":    "human_tRNA_chimeric_reads",
        "params":           TRNA_PARAMS,
    },
    {
        "label":            "ecoli_tRNA",
        "enabled":          True,
        "sequence_csv":     "ecoli_tRNA.csv",
        "interactions_csv": "stems2_ecoli_tRNA.csv",
        "output_prefix":    "ecoli_tRNA_chimeric_reads",
        "params":           TRNA_PARAMS,
    },
    {
        "label":            "human_rRNA",
        "enabled":          False,          # flip on once filenames confirmed
        "sequence_csv":     "human_rRNA.csv",
        "interactions_csv": "stems2_human_rRNA.csv",
        "output_prefix":    "human_rRNA_chimeric_reads",
        "params":           RRNA_PARAMS,
    },
    {
        "label":            "ecoli_rRNA",
        "enabled":          False,          # flip on once filenames confirmed
        "sequence_csv":     "ecoli_rRNA.csv",
        "interactions_csv": "stems2_ecoli_rRNA.csv",
        "output_prefix":    "ecoli_rRNA_chimeric_reads",
        "params":           RRNA_PARAMS,
    },
]

MAKE_CSV                       = True
MAKE_FASTA                     = True    # RNA alphabet (U), as in the original
MAKE_FASTQ                     = True    # DNA alphabet (T), for segemehl/RNAnue
MAKE_BED                       = True
MAKE_REFERENCE_FASTA_FOR_IGV   = True    # RNA alphabet (U)
MAKE_REFERENCE_FASTA_DNA       = True    # DNA alphabet (T), for segemehl index

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


def sample_fragment(rng:      random.Random,
                    seq_len:  int,
                    core:     Range1,
                    frag_len: int,
                    position: str,
                    p:        SimParams) -> Range1 | None:
    """
    Sample one fragment of length frag_len that overlaps core and keeps
    the ligation-facing end inside the core (junction-clean rule).

    position="first"  -> right end is the join; cannot go past core.end
    position="second" -> left end is the join;  cannot go before core.start

    Returns None if no valid placement exists.
    """
    if frag_len < 1:
        return None

    if position == "first":
        lo = max(1, core.start - p.max_arm_extension, core.start - frag_len + 1)
        hi = core.end - frag_len + 1
    elif position == "second":
        window_end = min(seq_len, core.end + p.max_arm_extension)
        lo = core.start
        hi = min(core.end, window_end - frag_len + 1)
    else:
        raise ValueError(f"position must be 'first' or 'second', got {position!r}")

    if lo > hi:
        return None

    start = rng.randint(lo, hi)
    return Range1(start, start + frag_len - 1)


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
                  p:                  SimParams) -> dict | None:
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

    # NOTE: forbidden-arm (cross-interaction overlap) checks intentionally
    # disabled -- matches the exact behaviour of the 97.7%-recall run.
    #if overlaps_forbidden_arm(first_frag,  arm_index, name, interaction_number, first_arm_id):
    #    return None
    #if overlaps_forbidden_arm(second_frag, arm_index, name, interaction_number, second_arm_id):
    #    return None

    read_seq = seq_slice(seq, first_frag) + seq_slice(seq, second_frag)

    frag1, frag2 = (first_frag, second_frag) if orientation == "arm1_to_arm2" else (second_frag, first_frag)

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

        if core1.end > seq_len or core2.end > seq_len:
            warnings.append(f"Skipping {name} int {interaction_number} -- arm exceeds length {seq_len}.")
            continue

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
                                     core1, core2, orientation, rng, arm_index, p)
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
        f"  arm_len drawn from [{lo}, {hi}] -> read lengths "
        f"{reads['read_len'].min()}..{reads['read_len'].max()} "
        f"(mean {reads['read_len'].mean():.1f})",
    ]

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
        output_prefix    = path_here(dataset["output_prefix"])

        validate_settings(p, label)

        if not sequence_csv.exists() or not interactions_csv.exists():
            print(f"Skipping {label} -- input files not found.")
            continue

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
            ref = path_here(f"{label}_reference_for_igv.fa")
            write_reference_fasta_and_fai(seq_df, ref, as_dna=False)
            print(f"  IGV ref FASTA: {ref}")
            print(f"  IGV ref index: {ref}.fai")

        if MAKE_REFERENCE_FASTA_DNA:
            ref_dna = path_here(f"{label}_reference_dna.fa")
            write_reference_fasta_and_fai(seq_df, ref_dna, as_dna=True)
            print(f"  DNA ref FASTA: {ref_dna}   (index with: segemehl.x -x {ref_dna.stem}.idx -d {ref_dna.name})")

        print(f"  {summarize(reads, p)}")

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
