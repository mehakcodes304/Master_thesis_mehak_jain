"""
Convert a stems CSV (sub_interaction_stem format) to a BED file
matching the format of human_tRNA_interactions.bed.

Each interaction produces two BED rows — one per arm — sharing the same color.
Arm 1 (left/lower)  → strand "+"
Arm 2 (right/higher) → strand "-"
Colors cycle through a palette of distinct hex colors per interaction number.

Chromosome names are built as "{name}_{id}" to match the reference FASTA
headers produced by csv_to_fasta.ipynb and generate_chimeric_reads_v2.py.
"""

import re
import pandas as pd
from pathlib import Path

COLORS = [
    "#e83337", "#02b4f6", "#f5a623", "#7ed321", "#9013fe",
    "#417505", "#d0021b", "#4a90d9", "#f8e71c", "#50e3c2",
    "#b8e986", "#bd10e0", "#9b9b9b", "#e94f77", "#00bfa5",
    "#ff6d00", "#304ffe", "#aa00ff", "#c6ff00", "#00e5ff",
]


def parse_stem_ranges(value: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Parse sub_interaction_stem into two (start, end) arm tuples.
    Supports range format "[(1-7), (66-72)]" and pair-list format.
    Always returns (lower-start arm, higher-start arm).
    """
    text = str(value)

    dash_matches = re.findall(r"\((\d+)\s*-\s*(\d+)\)", text)
    if len(dash_matches) >= 2:
        arms = [(int(a), int(b)) for a, b in dash_matches[:2]]
        return tuple(sorted(arms, key=lambda r: r[0]))

    pair_matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)
    if pair_matches:
        left  = [int(a) for a, _ in pair_matches]
        right = [int(b) for _, b in pair_matches]
        return tuple(sorted([(min(left), max(left)), (min(right), max(right))],
                             key=lambda r: r[0]))

    raise ValueError(f"Cannot parse: {value!r}")


def stems_to_bed(sequence_csv: Path,
                 input_csv:    Path,
                 output_bed:   Path,
                 track_name:   str,
                 track_desc:   str) -> None:
    """
    Convert a stems CSV to BED format.
    Requires the original sequence CSV to build name_id chromosome names
    that match the reference FASTA headers.
    """
    # Build name → id lookup so chrom names match the reference FASTA
    seq_df    = pd.read_csv(sequence_csv)[["name", "id"]]
    id_lookup = dict(zip(seq_df["name"], seq_df["id"]))

    df   = pd.read_csv(input_csv)
    rows = []

    for _, row in df.iterrows():
        name               = str(row["name"])
        interaction_number = int(row["interaction_number"])

        rna_id     = id_lookup.get(name, "")
        chrom_name = f"{name}_{rna_id}" if rna_id else name

        arm1, arm2 = parse_stem_ranges(row["sub_interaction_stem"])

        # BED: 0-based start, half-open end → subtract 1 from start only
        arm1_start, arm1_end = arm1[0] - 1, arm1[1]
        arm2_start, arm2_end = arm2[0] - 1, arm2[1]

        color   = COLORS[(interaction_number - 1) % len(COLORS)]
        cluster = f"cluster{interaction_number - 1}"

        rows.append([chrom_name, arm1_start, arm1_end, f"{cluster}_segment1",
                     0, "+", arm1_start, arm1_end, color])
        rows.append([chrom_name, arm2_start, arm2_end, f"{cluster}_segment2",
                     0, "-", arm2_start, arm2_end, color])

    output_bed.parent.mkdir(parents=True, exist_ok=True)
    with output_bed.open("w") as fh:
        fh.write(f'track name="{track_name}" description="{track_desc}" itemRgb="On"\n')
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")

    print(f"Written {len(rows)} BED rows ({len(rows)//2} interactions) → {output_bed}")


if __name__ == "__main__":
    stems_to_bed(
        sequence_csv = Path("human_tRNA.csv"),
        input_csv    = Path("stems2_human_tRNA.csv"),
        output_bed   = Path("human_tRNA_interactions.bed"),
        track_name   = "human tRNA RNA-RNA interactions",
        track_desc   = "Stem segments derived from human tRNA stem table",
    )


if __name__ == "__main__":
    stems_to_bed(
        sequence_csv = Path("human_rRNA.csv"),
        input_csv    = Path("stems2_human_rRNA.csv"),
        output_bed   = Path("human_rRNA_interactions.bed"),
        track_name   = "human rRNA RNA-RNA interactions",
        track_desc   = "Stem segments derived from human rRNA stem table",
    )


if __name__ == "__main__":
    stems_to_bed(
        sequence_csv = Path("ecoli_tRNA.csv"),
        input_csv    = Path("stems2_ecoli_tRNA.csv"),
        output_bed   = Path("ecoli_tRNA_interactions.bed"),
        track_name   = "ecoli tRNA RNA-RNA interactions",
        track_desc   = "Stem segments derived from ecoli tRNA stem table",
    )


if __name__ == "__main__":
    stems_to_bed(
        sequence_csv = Path("ecoli_rRNA.csv"),
        input_csv    = Path("stems2_ecoli_rRNA.csv"),
        output_bed   = Path("ecoli_rRNA_interactions.bed"),
        track_name   = "ecoli rRNA RNA-RNA interactions",
        track_desc   = "Stem segments derived from ecoli rRNA stem table",
    )