#!/usr/bin/env python3
"""
Generate simulated chimeric RNA reads from RNA interaction arms.

Run with:
    python generate_chimeric_reads_v2.py

Rules:
1. Keep RNA sequence text as given in the CSV, except uppercasing.
2. Do not reverse-complement.
3. Do not convert RNA to DNA or DNA to RNA.
4. Arm 1 and arm 2 have the same length within each read.
5. A fragment must not overlap another arm or another interaction arm.
6. The join has no unpaired bases between the two joined pieces.
7. Both orientations are supported inside the requested total per interaction.
8. BED12 output is written for IGV viewing.

Input coordinates are 1-based inclusive.
BED output uses standard BED coordinates (0-based start, half-open end).
"""

from __future__ import annotations

import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATASETS = [
    {
        "label":            "human_tRNA",
        "sequence_csv":     "human_tRNA.csv",
        "interactions_csv": "stems2_human_tRNA.csv",
        "output_prefix":    "human_tRNA_chimeric_reads1",
    },
    {
        "label":            "ecoli_tRNA",
        "sequence_csv":     "ecoli_tRNA.csv",
        "interactions_csv": "stems2_ecoli_tRNA.csv",
        "output_prefix":    "ecoli_tRNA_chimeric_reads1",
    },
]

READS_PER_INTERACTION = 200
ORIENTATION_MODE      = "both"    # "both" | "arm1_to_arm2" | "arm2_to_arm1"

MIN_READ_LEN      = 12
MAX_READ_LEN      = 50
MIN_ARM_LEN       = 6
MAX_ARM_LEN       = 25
MAX_ARM_EXTENSION    =100
MAX_JUNCTION_UNPAIRED = 4   # max unpaired bases allowed at the ligation junction
                             # 0 = strictly inside core (original junction-clean)
                             # 1-3 = realistic DDD: small loop/bulge at junction
                             # increase only if short cores produce 0 reads

MIN_INTRA_GAP  = None           # e.g. 10
MAX_INTRA_GAP  = None           # e.g. 80
MIN_INTER_GAP  = None           # e.g. 5
MAX_INTER_GAP  = None           # e.g. 50
INTER_GAP_MODE = "diff_mean"    # "diff_mean" | "diff_min" | "diff_max" | "primary_min"

UNIQUE_READS_PER_INTERACTION = False
MAX_ATTEMPTS_PER_READ        = 1000
SEED                         = 42

MAKE_CSV                       = True
MAKE_FASTA                     = True
MAKE_FASTQ                     = True
MAKE_BED                       = True
MAKE_REFERENCE_FASTA_FOR_IGV   = True

# Fill this list to convert existing read CSVs to BED without re-generating.
# Example: ["human_tRNA_chimeric_reads_200_per_interaction.csv"]
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
    """Remove whitespace and uppercase the sequence. changes U/T."""
    return str(seq).strip().replace(" ", "").replace("\n", "").upper().replace("U", "T")


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
      "[(1-7), (66-72)]"               ← range format
      "[(1, 72), (2, 71), (3, 70)]"    ← pair-list format
    """
    text = str(value)

    # range format: (a-b)
    dash_matches = re.findall(r"\((\d+)\s*-\s*(\d+)\)", text)
    if len(dash_matches) >= 2:
        arms = [Range1(int(a), int(b)) for a, b in dash_matches[:2]]
        return tuple(sorted(arms, key=lambda r: r.start))  # type: ignore[return-value]

    # pair-list format: (a, b)
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
        for idx, row in enumerate(rows[:-1]):           # skip the last — it has no next
            key = (row.name, row.interaction_number)
            lookup[key] = parse_stem_ranges(rows[idx + 1].sub_interaction_stem)

    return lookup


def build_arm_index(interactions: pd.DataFrame) -> dict[str, list[ArmRecord]]:
    """
    Build a dict: RNA name → list of ArmRecords for that RNA.

    Indexing by name means overlap checks only scan arms on the same RNA —
    much faster than scanning all arms globally.
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
    diff_* modes use pre-computed columns; primary_min computes from arm ranges.
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
        raise ValueError("ORIENTATION_MODE must be both, arm1_to_arm2, or arm2_to_arm1")
    half = total // 2
    return {"arm1_to_arm2": half, "arm2_to_arm1": total - half}


def choose_arm_length(rng: random.Random) -> int | None:
    """
    Pick one arm length so that 2 × arm_len satisfies both read-length
    and arm-length bounds. Returns None if no valid length exists.
    """
    lo = max(MIN_ARM_LEN, math.ceil(MIN_READ_LEN / 2))
    hi = min(MAX_ARM_LEN, MAX_READ_LEN // 2)
    return rng.randint(lo, hi) if lo <= hi else None


def sample_fragment(rng:      random.Random,
                    seq_len:  int,
                    core:     Range1,
                    frag_len: int,
                    position: str) -> Range1 | None:
    """
    Sample one fragment of length frag_len that:
      1. Overlaps the core arm by at least 1 nucleotide
      2. Has its ligation-facing (junction) end within MAX_JUNCTION_UNPAIRED
         bases of the core boundary — allowing a small number of unpaired
         bases at the chimeric ligation point, as seen in real DDD/CLASH data.

    position="first"  → right end is the junction end (faces arm2)
                        right end can go at most MAX_JUNCTION_UNPAIRED past core.end
                        left end can extend freely up to MAX_ARM_EXTENSION
    position="second" → left end is the junction end (faces arm1)
                        left end can go at most MAX_JUNCTION_UNPAIRED before core.start
                        right end can extend freely up to MAX_ARM_EXTENSION

    Returns None if no valid placement exists.
    """
    if frag_len < 1:
        return None

    window_start = max(1,       core.start - MAX_ARM_EXTENSION)
    window_end   = min(seq_len, core.end   + MAX_ARM_EXTENSION)

    if position == "first":
        # Left end: free extension upstream
        # Right end (junction): at most MAX_JUNCTION_UNPAIRED past core.end
        junction_max_end = core.end + MAX_JUNCTION_UNPAIRED
        lo = max(window_start, core.start - frag_len + 1)
        hi = min(window_end - frag_len + 1,
                 junction_max_end - frag_len + 1,
                 core.end)          # must still overlap core by >= 1 nt

    elif position == "second":
        # Right end: free extension downstream
        # Left end (junction): at most MAX_JUNCTION_UNPAIRED before core.start
        junction_min_start = core.start - MAX_JUNCTION_UNPAIRED
        lo = max(window_start,
                 junction_min_start,
                 core.start - frag_len + 1)  # must still overlap core by >= 1 nt
        hi = min(window_end - frag_len + 1, core.end)

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
    Uses the pre-built name index so only relevant arms are scanned.
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
                  arm_index:          dict[str, list[ArmRecord]]) -> dict | None:
    """
    Attempt to sample one valid chimeric read for one interaction/orientation.
    Returns None if no valid placement can be found with this arm length draw.
    """
    arm_len = choose_arm_length(rng)
    if arm_len is None:
        return None

    # Assign first/second piece based on orientation
    if orientation == "arm1_to_arm2":
        first_arm_id,  second_arm_id  = 1, 2
        first_core,    second_core    = core1, core2
    else:
        first_arm_id,  second_arm_id  = 2, 1
        first_core,    second_core    = core2, core1

    first_frag  = sample_fragment(rng, seq_len, first_core,  arm_len, "first")
    second_frag = sample_fragment(rng, seq_len, second_core, arm_len, "second")

    if first_frag is None or second_frag is None:
        return None

    if first_frag.overlaps(second_frag):
        return None
    
    #if overlaps_forbidden_arm(first_frag,  arm_index, name, interaction_number, first_arm_id):
    #    return None
    #if overlaps_forbidden_arm(second_frag, arm_index, name, interaction_number, second_arm_id):
    #    return None

    read_seq = seq_slice(seq, first_frag) + seq_slice(seq, second_frag)

    # Normalise frag1/frag2 to always mean arm1/arm2 regardless of orientation
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


def generate_reads_for_dataset(sequence_csv:    Path,
                                interactions_csv: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Full pipeline for one dataset:
      1. Load and validate inputs
      2. Build lookup structures
      3. For each interaction and orientation, sample the target number of reads
      4. Return (reads DataFrame, warnings, sequence DataFrame)
    """
    rng = random.Random(SEED)
    seq_df, interactions = load_inputs(sequence_csv, interactions_csv)

    # Build lookup by full name AND by short name (stripped of _URSxxxxxxxx_taxid suffix).
    # This handles CSVs where name is already "rRNA_5S_1_8_URS00000F9D45_9606"
    # while the interactions CSV uses just "rRNA_5S_1_8".
    seq_by_name: dict[str, dict] = {}
    for entry in seq_df.to_dict("records"):
        full_name  = str(entry["name"])
        short_name = re.sub(r"_URS[A-Z0-9]+_\d+$", "", full_name)
        seq_by_name[full_name]  = entry   # always register the full name
        if short_name != full_name:
            seq_by_name[short_name] = entry   # also register the short name
    next_lookup  = build_next_range_lookup(interactions)
    arm_index    = build_arm_index(interactions)            # indexed by RNA name for fast lookup

    records:  list[dict] = []
    warnings: list[str]  = []
    warned_missing: set[str] = set()

    interactions = interactions.sort_values(["name", "interaction_number"]).reset_index(drop=True)

    for _, row in interactions.iterrows():
        name               = str(row["name"])
        interaction_number = int(row["interaction_number"])

        if name not in seq_by_name:
            if name not in warned_missing:
                warnings.append(f"Skipping {name} — sequence not found.")
                warned_missing.add(name)
            continue

        seq_meta = seq_by_name[name]
        seq      = str(seq_meta["seq"])
        seq_len  = len(seq)
        core1, core2 = parse_stem_ranges(row["sub_interaction_stem"])

        if core1.end > seq_len or core2.end > seq_len:
            warnings.append(f"Skipping {name} int {interaction_number} — arm exceeds length {seq_len}.")
            continue

        intra_gap   = interval_distance(core1, core2)
        next_ranges = next_lookup.get((name, interaction_number))
        inter_gap   = get_interaction_gap(row, next_ranges, (core1, core2), INTER_GAP_MODE)

        if not within_bounds(intra_gap, MIN_INTRA_GAP, MAX_INTRA_GAP):
            continue
        if not within_bounds(inter_gap, MIN_INTER_GAP, MAX_INTER_GAP):
            continue


        seen: set[tuple] = set()
        targets = orientation_targets(READS_PER_INTERACTION, ORIENTATION_MODE)

        # Simple counter — avoids O(n²) scan of records list on every emitted read
        read_count_this_interaction = 0

        for orientation, target_count in targets.items():
            emitted  = 0
            attempts = 0
            max_attempts = target_count * MAX_ATTEMPTS_PER_READ

            while emitted < target_count and attempts < max_attempts:
                attempts += 1

                read = make_one_read(seq, seq_len, name, interaction_number,
                                     core1, core2, orientation, rng, arm_index)
                if read is None:
                    continue

                # Uniqueness check: coordinates + orientation (no need to store full sequence)
                signature = (orientation,
                             read["frag_arm1_start"], read["frag_arm1_end"],
                             read["frag_arm2_start"], read["frag_arm2_end"])
                if UNIQUE_READS_PER_INTERACTION and signature in seen:
                    continue
                seen.add(signature)

                emitted += 1
                read_count_this_interaction += 1

                # igv_name = name_id combined — used as chrom in BED and reference FASTA
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
                    "inter_gap_mode":       INTER_GAP_MODE,
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
    """Write generated reads as FASTA."""
    with output_fasta.open("w", encoding="utf-8") as fh:
        for row in reads.itertuples(index=False):
            fh.write(f">{row.read_id}\n{row.sequence}\n")

def write_fastq(reads: pd.DataFrame, output_fastq: Path) -> None:
    """
    Write generated reads as FASTQ with uniform Phred 40 quality scores.
    Phred 40 (ASCII 'I') is standard for simulated reads — indicates perfect
    base quality since reads are generated without sequencing errors.
    Sequences are already in DNA alphabet (U->T applied by clean_seq).
    """
    with output_fastq.open("w", encoding="utf-8") as fh:
        for row in reads.itertuples(index=False):
            qual = "I" * len(row.sequence)
            fh.write(f"@{row.read_id}\n{row.sequence}\n+\n{qual}\n")


def write_reference_fasta_and_fai(seq_df: pd.DataFrame, output_fasta: Path) -> None:
    """
    Write RNA sequences as a FASTA reference and matching .fai index for IGV.
    Opens in binary mode so byte offsets in the FAI are always correct.
    """
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    fai_rows: list[tuple] = []
    offset = 0

    with output_fasta.open("wb") as fh:
        for row in seq_df.itertuples(index=False):
            name = f"{row.name}_{row.id}" if hasattr(row, "id") and str(row.id) else str(row.name)
            seq  = str(row.seq)

            header   = f">{name}\n".encode()
            seq_line = f"{seq}\n".encode()

            fh.write(header)
            seq_offset = offset + len(header)   # byte position of first base
            fh.write(seq_line)

            fai_rows.append((name, len(seq), seq_offset, len(seq), len(seq_line)))
            offset += len(header) + len(seq_line)

    fai_path = output_fasta.with_suffix(output_fasta.suffix + ".fai")
    with fai_path.open("w", encoding="utf-8") as fh:
        for row in fai_rows:
            fh.write("\t".join(map(str, row)) + "\n")


def bed12_row_from_read(row: pd.Series) -> list:
    """Convert one read row to a BED12 record. Uses igv_name as chrom."""
    arm1   = Range1(int(row["frag_arm1_start"]), int(row["frag_arm1_end"]))
    arm2   = Range1(int(row["frag_arm2_start"]), int(row["frag_arm2_end"]))
    blocks = sorted([arm1, arm2], key=lambda r: r.start)

    chrom_start   = blocks[0].start - 1        # BED is 0-based
    chrom_end     = blocks[-1].end
    block_sizes   = ",".join(str(b.length)                        for b in blocks) + ","
    block_starts  = ",".join(str(b.start - 1 - chrom_start)      for b in blocks) + ","
    item_rgb      = "0,0,255" if row.get("orientation") == "arm1_to_arm2" else "255,0,0"
    chrom         = str(row.get("igv_name", row["name"]))   # use igv_name if present

    return [
        chrom, chrom_start, chrom_end, str(row["read_id"]),
        0, "+", chrom_start, chrom_end, item_rgb, 2, block_sizes, block_starts,
    ]


def compute_coverage(reads: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-base arm coverage across all chimeric reads.
    Each read contributes two arm fragments (arm1 and arm2).
    Uses a difference-array approach for efficiency.

    Returns a long-format DataFrame with columns:
        chrom, start (0-based), end (half-open), coverage
    """
    from collections import defaultdict

    # diff[chrom][pos] += delta — increment at start, decrement after end
    diff: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for row in reads.itertuples(index=False):
        chrom = str(getattr(row, "igv_name", row.name))
        for start_col, end_col in [("frag_arm1_start", "frag_arm1_end"),
                                    ("frag_arm2_start", "frag_arm2_end")]:
            s = int(getattr(row, start_col)) - 1   # convert to 0-based
            e = int(getattr(row, end_col))          # half-open end
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
        raise ValueError(f"Cannot write BED — missing columns: {sorted(missing)}")

    output_bed.parent.mkdir(parents=True, exist_ok=True)
    with output_bed.open("w", encoding="utf-8") as fh:
        fh.write('track name="chimeric_reads" description="Simulated chimeric RNA reads" '
                 'visibility=2 itemRgb="On"\n')
        for _, row in reads.iterrows():
            fh.write("\t".join(map(str, bed12_row_from_read(row))) + "\n")


# =============================================================================
# VALIDATION + RUN LOOP
# =============================================================================

def validate_settings() -> None:
    """Catch impossible setting combinations before doing any work."""
    if READS_PER_INTERACTION < 1:
        raise ValueError("READS_PER_INTERACTION must be at least 1")
    if MIN_READ_LEN < 1 or MAX_READ_LEN < MIN_READ_LEN:
        raise ValueError("Read length bounds are invalid")
    if MIN_ARM_LEN < 1 or MAX_ARM_LEN < MIN_ARM_LEN:
        raise ValueError("Arm length bounds are invalid")
    if MIN_READ_LEN > 2 * MAX_ARM_LEN:
        raise ValueError("MIN_READ_LEN is larger than 2 × MAX_ARM_LEN")
    if MAX_READ_LEN < 2 * MIN_ARM_LEN:
        raise ValueError("MAX_READ_LEN is smaller than 2 × MIN_ARM_LEN")
    if MAX_ARM_EXTENSION < 0:
        raise ValueError("MAX_ARM_EXTENSION must be >= 0")
    if ORIENTATION_MODE not in {"both", "arm1_to_arm2", "arm2_to_arm1"}:
        raise ValueError("ORIENTATION_MODE must be both, arm1_to_arm2, or arm2_to_arm1")
    if INTER_GAP_MODE not in {"diff_mean", "diff_min", "diff_max", "primary_min"}:
        raise ValueError("INTER_GAP_MODE must be diff_mean, diff_min, diff_max, or primary_min")


def summarize(reads: pd.DataFrame) -> str:
    """One-line summary of a generated reads DataFrame."""
    if reads.empty:
        return "no reads generated"
    counts = reads.groupby(["name", "interaction_number"]).size()
    orientations = reads.groupby("orientation").size().to_dict() if "orientation" in reads else {}
    equal = bool((reads["arm1_len"] == reads["arm2_len"]).all())
    return (
        f"{len(reads)} reads over {len(counts)} interactions — "
        f"min/max per interaction: {counts.min()}/{counts.max()} — "
        f"orientations: {orientations} — equal arm lengths: {equal}"
    )


def run_generation() -> list[str]:
    """Generate reads for every configured dataset and write all requested outputs."""
    all_warnings: list[str] = []

    for dataset in DATASETS:
        label            = str(dataset["label"])
        sequence_csv     = path_here(dataset["sequence_csv"])
        interactions_csv = path_here(dataset["interactions_csv"])
        output_prefix    = path_here(dataset["output_prefix"])

        if not sequence_csv.exists() or not interactions_csv.exists():
            print(f"Skipping {label} — input files not found.")
            continue

        print(f"Processing {label}")
        reads, warnings, seq_df = generate_reads_for_dataset(sequence_csv, interactions_csv)
        all_warnings.extend(f"{label}: {w}" for w in warnings)

        if MAKE_CSV:
            p = output_prefix.with_suffix(".csv")
            reads.to_csv(p, index=False)
            print(f"  CSV:   {p}")

        if MAKE_FASTA:
            if reads.empty:
                print(f"  FASTA: skipped — no reads generated for {label}")
            else:
                p = output_prefix.with_suffix(".fa")
                write_fasta(reads, p)
                print(f"  FASTA: {p}")

        if MAKE_FASTQ:
            if reads.empty:
                print(f"  FASTQ: skipped — no reads generated for {label}")
            else:
                p = output_prefix.with_suffix(".fastq")
                write_fastq(reads, p)
                print(f"  FASTQ: {p}")

        if MAKE_BED:
            if reads.empty:
                print(f"  BED12: skipped — no reads generated for {label}")
            else:
                p = output_prefix.with_suffix(".bed")
                write_bed12(reads, p)
                print(f"  BED12: {p}")

                # Coverage files
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
            write_reference_fasta_and_fai(seq_df, ref)
            print(f"  IGV ref FASTA: {ref}")
            print(f"  IGV ref index: {ref}.fai")

        print(f"  {summarize(reads)}")

    return all_warnings


def run_existing_csv_to_bed() -> None:
    """Convert any existing read CSVs listed in EXISTING_READ_CSVS_TO_BED to BED12."""
    for csv_name in EXISTING_READ_CSVS_TO_BED:
        csv_path = path_here(csv_name)
        if not csv_path.exists():
            print(f"Skipping BED conversion — not found: {csv_path}")
            continue
        bed_path = csv_path.with_suffix(".bed")
        write_bed12(pd.read_csv(csv_path), bed_path)
        print(f"Converted {csv_path.name} → {bed_path.name}")


def main() -> int:
    validate_settings()
    warnings = run_generation()
    run_existing_csv_to_bed()

    if warnings:
        print("\nWarnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())