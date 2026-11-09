#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: $0 <dataset>  e.g. ecoli_tRNA or human_tRNA}"

BASE=/mnt/ceph/mjain/software/envs/bio
STEMS_CSV="$BASE/stems/${DATASET}/stems2_${DATASET}.csv"
MATRIX_DIR="$BASE/contact_matrices/${DATASET}"
PLOT_DIR="$MATRIX_DIR/plots"
SCRIPT="$BASE/plot_contact_heatmap.py"
CHIMERIC="$BASE/chimeric_reads/${DATASET}/${DATASET}_chimeric_reads.csv"

mkdir -p "$PLOT_DIR"

python - "$STEMS_CSV" "$CHIMERIC" <<'PYEOF'
import sys, csv, subprocess, glob, os

csv_path   = sys.argv[1]
chim_path  = sys.argv[2]
dataset    = os.path.basename(os.path.dirname(csv_path))
base       = os.path.dirname(os.path.dirname(os.path.dirname(csv_path)))
matrix_dir = os.path.join(base, "contact_matrices", dataset)
plot_dir   = os.path.join(matrix_dir, "plots")
script     = os.path.join(base, "plot_contact_heatmap.py")

LARGE_NT = 500  # skip per-stem zooms above this

seen = set()
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name   = row["name"].strip()
        length = int(row["seq_len"].strip())
        if name in seen:
            continue
        seen.add(name)

        pattern = os.path.join(matrix_dir, f"{name}_URS*_contact_matrix_sparse.tsv")
        matches = glob.glob(pattern)
        if not matches:
            print(f"  SKIP {name} — no sparse matrix found")
            continue

        cmd = [
            "python", script,
            "--sparse",  matches[0],
            "--stems",   csv_path,
            "--rna",     name,
            "--length",  str(length),
            "--out",     plot_dir,
        ]
        if os.path.exists(chim_path):
            cmd += ["--chimeric", chim_path]
        if length > LARGE_NT:
            cmd += ["--no_zoom", "--min_support", "5"]

        print(f"  Plotting {name} (len={length}) ...")
        subprocess.run(cmd, check=True)

print("Done.")
PYEOF