#!/usr/bin/env python3
"""
Iteratively optimise RNAnue params.cfg to maximise recovery of known stem
interactions.

Strategy: COORDINATE DESCENT.
  - Start from the base config.
  - Sweep one parameter at a time over a small list of candidate values.
  - Keep the value that scores best, then move to the next parameter.
  - Optionally repeat the whole pass until nothing improves.

Efficiency: each parameter is tagged with the EARLIEST pipeline stage it
affects. When testing a value we only rerun RNAnue from that stage onward:
  - analyze-stage params (mincount, clustfrac, maxselfoverlap, padj) -> rerun analyze only (fast)
  - detect-stage params  -> rerun detect + analyze
  - align-stage params   -> rerun align + detect + analyze
  - preprocess params    -> rerun everything
Cheap stages are swept first so most of the search is fast.

Every trial is logged to optimization_log.csv, and the best config is written
to params_best.cfg.
"""

import subprocess
import os
import re
import shutil
import datetime
from pathlib import Path
import pandas as pd

# ============================================================================
# USER CONFIGURATION  —  edit these paths for your dataset
# ============================================================================
RNANUE       = "/mnt/ceph/mjain/software/envs/bio/RNAnue/build/release/RNAnue"
WORKDIR      = "/mnt/ceph/mjain/software/envs/bio/rnanue_run/human_rRNA"   # dir containing params.cfg
BASE_CONFIG  = f"{WORKDIR}/params.cfg"
KNOWN_STEMS  = "/mnt/ceph/mjain/software/envs/bio/stems2_human_rRNA.csv"
INTERACTIONS = f"{WORKDIR}/output/04_analyze/treatment/condition1/condition1_interactions.tsv"
LD_LIB       = "/mnt/ceph/mjain/software/envs/bio/lib"   # for gnuplot/libtiff

TOLERANCE    = 10          # nt tolerance when matching detected -> known
OBJECTIVE    = "recall"    # "recall" | "f1" | "precision" — used only when
                           # TARGET_RECALL is None
PASSES       = 2           # how many full coordinate-descent passes to run

# --- Target-driven (constrained) optimisation -------------------------------
# Set a recall target (e.g. 0.9 or 1.0) to switch from "maximise a metric" to
# "reach this recall, then keep the cleanest config that does".
#
# Ranking rule:
#   recall <  TARGET_RECALL  ->  score = recall / TARGET_RECALL          (< 1)
#                                ranks by how close it got to the target
#   recall >= TARGET_RECALL  ->  score = 1 + precision                   (>= 1)
#                                target met; now ranked by precision
#
# So ANY config meeting the target always beats any config that doesn't, and
# among those that meet it the one with fewest false positives wins.
# Set to None to disable and fall back to plain OBJECTIVE maximisation.
TARGET_RECALL = 1.0

# Stop as soon as a config meets the target with precision at or above this.
# Set to None to always finish every pass (recommended — lets you see the full
# trade-off curve in the log rather than stopping at the first acceptable hit).
STOP_WHEN_PRECISION_AT_LEAST = None

# Parameters to sweep. Each entry: name -> (stage, [candidate values]).
# Order matters only for readability; the script sorts by stage cost itself.
# Start with the analyze-stage knobs (cheap + they control merging/splitting
# of clusters, which is exactly what governs recovery).
PARAM_GRID = {
    "mincount":       ("analyze", [3, 5, 10]),
    "clustfrac":      ("analyze", [0.3, 0.5, 0.8]),
    "maxselfoverlap": ("analyze", [0.1, 0.5, 1.0]),
    "padj":           ("analyze", [1.0]),
    "minsplicecov":   ("align",   [40, 60]),
    "minfragsco":     ("align",   [4, 5]),
    "accuracy":       ("align",   [85, 90, 95]),
}

STAGE_ORDER = ["preprocess", "align", "detect", "analyze"]

# ============================================================================
# CONFIG FILE HANDLING
# ============================================================================
def read_config(path):
    """Read params.cfg into a list of raw lines (structure preserved)."""
    with open(path) as f:
        return f.readlines()


def write_config(base_lines, overrides, out_path):
    """
    Write a new config: for every key in `overrides`, replace its value line;
    keep all other lines (comments, blank lines) untouched.
    """
    keys_written = set()
    out = []
    for line in base_lines:
        m = re.match(r"^(\s*)([A-Za-z_]+)(\s*)=(\s*)(.*)$", line)
        if m and m.group(2) in overrides:
            key = m.group(2)
            out.append(f"{key} = {overrides[key]}\n")
            keys_written.add(key)
        else:
            out.append(line)
    # Append any override keys that weren't already present
    for key, val in overrides.items():
        if key not in keys_written:
            out.append(f"{key} = {val}\n")
    with open(out_path, "w") as f:
        f.writelines(out)


# ============================================================================
# RUNNING RNANUE
# ============================================================================
def run_stages(config_path, from_stage):
    """Run RNAnue from `from_stage` through analyze. Returns True on success."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    start = STAGE_ORDER.index(from_stage)
    for stage in STAGE_ORDER[start:]:
        proc = subprocess.run(
            [RNANUE, stage, "--config", config_path],
            cwd=WORKDIR, env=env,
            capture_output=True, text=True,
        )
        # gnuplot may print errors but still produce data; only treat a missing
        # interactions file (checked later) as a real failure.
        if proc.returncode != 0 and stage == "analyze":
            # analyze sometimes returns non-zero on the cosmetic gnuplot crash
            pass
    return Path(INTERACTIONS).exists()


# ============================================================================
# RECOVERY SCORING  (same logic as check_recovery.py)
# ============================================================================
def parse_arms(val):
    m = re.findall(r"\((\d+)-(\d+)\)", str(val))
    if len(m) >= 2:
        arms = sorted([(int(a), int(b)) for a, b in m[:2]], key=lambda x: x[0])
        return arms[0], arms[1]
    m = re.findall(r"\((\d+)\s*,\s*(\d+)\)", str(val))
    if m:
        left  = [int(a) for a, _ in m]
        right = [int(b) for _, b in m]
        return (min(left), max(left)), (min(right), max(right))
    return None, None


def overlaps(a_s, a_e, b_s, b_e, tol):
    return a_s <= b_e + tol and b_s <= a_e + tol


def evaluate(interactions_path, known_path, tol):
    """Return (recall, precision, f1, tp, fp, fn) for one RNAnue result.

    If the interactions file is missing, empty, or header-only (which happens
    when a strict parameter combination detects nothing), this returns all
    zeros rather than crashing — that trial simply recovered nothing.
    """
    known = pd.read_csv(known_path)
    n_known = len(known)

    # Guard: file missing or unreadable
    try:
        det = pd.read_csv(interactions_path, sep="\t")
    except Exception:
        return 0.0, 0.0, 0.0, 0, 0, n_known

    # Guard: no data rows, or expected columns absent
    required = {"fst_feat_id", "sec_feat_id",
                "fst_seg_strt", "fst_seg_end",
                "sec_seg_strt", "sec_seg_end"}
    if det.empty or not required.issubset(det.columns):
        return 0.0, 0.0, 0.0, 0, 0, n_known

    det["intra"] = det["fst_feat_id"] == det["sec_feat_id"]
    intra = det[det["intra"]]
    inter = det[~det["intra"]]

    matched = set()
    wrong_intra = 0
    for _, row in intra.iterrows():
        name = str(row["fst_feat_id"]).replace("gene_", "")
        fs, fe = int(row["fst_seg_strt"]), int(row["fst_seg_end"])
        ss, se = int(row["sec_seg_strt"]), int(row["sec_seg_end"])
        found = False
        for _, sr in known[known["name"] == name].iterrows():
            a1, a2 = parse_arms(sr["sub_interaction_stem"])
            if not a1:
                continue
            if (overlaps(fs, fe, a1[0], a1[1], tol) and overlaps(ss, se, a2[0], a2[1], tol)) or \
               (overlaps(fs, fe, a2[0], a2[1], tol) and overlaps(ss, se, a1[0], a1[1], tol)):
                matched.add((name, int(sr["interaction_number"])))
                found = True
                break
        if not found:
            wrong_intra += 1

    tp = len(matched)
    fn = len(known) - tp
    fp = wrong_intra + len(inter)
    total_det = len(det)

    recall    = tp / len(known) if len(known) else 0.0
    precision = tp / total_det  if total_det   else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return recall, precision, f1, tp, fp, fn


def objective_value(recall, precision, f1):
    """
    Score a trial.

    If TARGET_RECALL is set, this is a CONSTRAINED objective:
      - below target: score < 1, ranked by progress toward the target
      - at/above target: score >= 1, ranked by precision
    so meeting the recall target always dominates, and precision breaks ties.

    If TARGET_RECALL is None, fall back to maximising OBJECTIVE.
    """
    if TARGET_RECALL is not None:
        if recall >= TARGET_RECALL:
            return 1.0 + precision          # target met -> rank by precision
        return recall / TARGET_RECALL       # not met  -> rank by closeness (<1)
    return {"recall": recall, "precision": precision, "f1": f1}[OBJECTIVE]


def target_met(recall):
    return TARGET_RECALL is not None and recall >= TARGET_RECALL


# ============================================================================
# OPTIMISER  (coordinate descent)
# ============================================================================
def optimise():
    base_lines = read_config(BASE_CONFIG)
    tmp_config = f"{WORKDIR}/params_trial.cfg"

    # Sort parameters cheapest-stage-first so most trials are fast
    stage_cost = {s: i for i, s in enumerate(STAGE_ORDER)}
    params_sorted = sorted(PARAM_GRID.items(),
                           key=lambda kv: -stage_cost[kv[1][0]])  # analyze(3) first

    current = {}                       # best value chosen so far per param
    log_rows = []
    best_score = -1.0
    best_overrides = {}

    # Live log paths — written after every single trial so nothing is lost
    log_csv   = f"{WORKDIR}/optimization_log.csv"
    log_xlsx  = f"{WORKDIR}/optimization_log.xlsx"
    trial_num = 0

    # Fresh CSV with a header row
    log_columns = ["trial", "timestamp", "pass", "param", "value", "stage",
                   "recall", "precision", "f1", "tp", "fp", "fn",
                   "objective", "score", "target_recall", "target_met",
                   "is_best_so_far", "config_snapshot"]
    with open(log_csv, "w") as f:
        f.write(",".join(log_columns) + "\n")

    for pass_i in range(1, PASSES + 1):
        improved = False
        print(f"\n===== PASS {pass_i} =====")

        for param, (stage, values) in params_sorted:
            best_val_score = -1.0
            best_val = current.get(param)

            for val in values:
                overrides = dict(current)
                overrides[param] = val
                write_config(base_lines, overrides, tmp_config)

                ok = run_stages(tmp_config, stage)
                if not ok:
                    print(f"  {param}={val}: FAILED (no interactions file)")
                    continue

                recall, precision, f1, tp, fp, fn = evaluate(
                    INTERACTIONS, KNOWN_STEMS, TOLERANCE)
                score = objective_value(recall, precision, f1)

                trial_num += 1
                is_best = score > best_score
                # Snapshot of the full config used for this trial (all current
                # best values plus the one being tested)
                snapshot = ";".join(f"{k}={v}" for k, v in overrides.items())

                row = {
                    "trial": trial_num,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pass": pass_i, "param": param, "value": val, "stage": stage,
                    "recall": round(recall, 4), "precision": round(precision, 4),
                    "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
                    "objective": ("target_recall" if TARGET_RECALL is not None
                                  else OBJECTIVE),
                    "score": round(score, 4),
                    "target_recall": TARGET_RECALL if TARGET_RECALL is not None else "",
                    "target_met": target_met(recall),
                    "is_best_so_far": is_best,
                    "config_snapshot": snapshot,
                }
                log_rows.append(row)

                # Append this trial to the CSV immediately (crash-safe)
                with open(log_csv, "a") as f:
                    f.write(",".join(str(row[c]) for c in log_columns) + "\n")

                # Refresh the Excel copy after every trial too
                try:
                    pd.DataFrame(log_rows).to_excel(log_xlsx, index=False)
                except Exception as e:
                    pass  # openpyxl may be missing; CSV is the source of truth

                flag = ""
                if target_met(recall):
                    flag = "  [TARGET MET]"
                if is_best:
                    flag += "  <-- best"
                print(f"  [{trial_num}] {param}={val:<6} recall={recall:.3f} "
                      f"precision={precision:.3f} f1={f1:.3f}{flag}")

                if score > best_val_score:
                    best_val_score = score
                    best_val = val

            # Lock in the best value for this parameter
            if best_val is not None:
                current[param] = best_val
                print(f"  -> best {param} = {best_val} ({OBJECTIVE}={best_val_score:.3f})")
                if best_val_score > best_score:
                    best_score = best_val_score
                    best_overrides = dict(current)
                    improved = True

        if not improved:
            print("No improvement this pass — stopping early.")
            break

    # Write the winning config. The log CSV/XLSX are already fully written.
    write_config(base_lines, best_overrides, f"{WORKDIR}/params_best.cfg")
    try:
        pd.DataFrame(log_rows).to_excel(log_xlsx, index=False)
    except Exception:
        pass

    print("\n===== BEST CONFIG =====")
    for k, v in best_overrides.items():
        print(f"  {k} = {v}")

    if log_rows:
        df = pd.DataFrame(log_rows)
        best_row = df.loc[df["score"].idxmax()]
        print(f"\n  recall    = {best_row['recall']:.3f}")
        print(f"  precision = {best_row['precision']:.3f}")
        print(f"  f1        = {best_row['f1']:.3f}")

        if TARGET_RECALL is not None:
            n_met = int(df["target_met"].sum())
            if n_met:
                met = df[df["target_met"]]
                best_met = met.loc[met["precision"].idxmax()]
                print(f"\n  TARGET RECALL {TARGET_RECALL} MET by {n_met} of "
                      f"{len(df)} trials.")
                print(f"  Best precision among them: {best_met['precision']:.3f} "
                      f"(trial {int(best_met['trial'])})")
                print(f"  Settings: {best_met['config_snapshot']}")
            else:
                best_recall = df["recall"].max()
                print(f"\n  TARGET RECALL {TARGET_RECALL} NOT REACHED.")
                print(f"  Best recall achieved: {best_recall:.3f}")
                print("  Config parameters alone cannot close this gap — the "
                      "limit is upstream (read generation / alignment).")
                print("  Next step: optimise the read generator "
                      "(optimize_generator.py), not params.cfg.")
    print(f"\nWritten: {WORKDIR}/params_best.cfg")
    print(f"Log CSV:  {WORKDIR}/optimization_log.csv")
    print(f"Log XLSX: {WORKDIR}/optimization_log.xlsx")


if __name__ == "__main__":
    optimise()