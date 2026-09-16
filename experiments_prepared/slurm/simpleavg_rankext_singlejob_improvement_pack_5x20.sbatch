#!/bin/bash
# =============================================================================
# SIMPLEAVG + RANKEXT JOINT SINGLE-JOB IMPROVEMENT PACK -- 5x20, seed 42.
# Prepared 2026-09-15/16. NOT SUBMITTED. Do not `sbatch` this without explicit
# author review -- see thesis_agent/reports/
# simpleavg_rankext_singlejob_improvement_pack_final_readiness.md.
#
# SUPERSEDES the earlier SimpleAvg-only launcher
# (simpleavg_singlejob_improvement_pack_5x20.sbatch) -- that one targets the
# obsolete SimpleAvg-only script and must NOT be used for the final run.
#
# DESIGN: because only ONE practical cluster allocation is available, this
# launcher runs the SAME allocation in TWO phases:
#   PHASE 1 -- a genuinely tiny GPU "preflight" smoke test of the exact same
#              script, with FAST_RUN_DEBUG=1 (reduced train/val/test image
#              counts and epoch counts ONLY -- protocol shape, ranks,
#              schedules, and all scientific logic are byte-identical to the
#              real run; see the script's own "if FAST_RUN" branches). Output
#              goes to a completely separate root (results_debug_preflight/,
#              never results/), so it can never be mistaken for or resumed
#              as real scientific output.
#   GATE    -- the preflight's own result CSV is verified structurally:
#              exactly 14 rows, 5 SimpleAvg training methods each with both
#              {arithmetic, do_merge}, 4 RankExt training methods each with
#              {persistent}, no duplicate rows, no missing accuracy values.
#   PHASE 2 -- ONLY if the gate passes: the real run (FAST_RUN unset/0),
#              9 training arms x 6 epochs, 14 evaluated configurations,
#              real output under results/.
# If Phase 1 fails (non-zero exit) OR the gate fails any check, Phase 2 is
# NEVER started and the job exits non-zero.
#
# One cluster job, 9 training arms (5 SimpleAvg + 4 RankExt), 14 evaluated
# configurations (SimpleAvg: 5 arms x {arithmetic, do_merge}; RankExt: 4 arms
# x {persistent}), 6 epochs for every arm in Phase 2 (intentional protocol
# change from Exp1's 9 epochs -- see the target script's own LORA_EPOCHS/
# RANKEXT_EPOCHS comment). SimpleAvg rank=80/alpha=160/scaling=2; RankExt
# schedule [16,32,48,64,80]/scaling=2 (Exp2's [32,64,96,128,160] is never
# active -- this is NOT a capacity ablation). Persistence/resume is built in
# (see the script's own CHECKPOINTS_DIR, isolated per RUN_NAME_BASE, stable
# across restarts) -- a crash/timeout mid-Phase-2 does not require retraining
# already-completed arms on resubmission (resubmitting would re-run the
# preflight too, which is cheap -- a few minutes -- not a concern).
#
# No conda/module activation line is added here: the python interpreter is
# invoked via its absolute path (CLUSTER_PYTHON below), so no shell env
# activation step is needed to put the right interpreter on PATH.
# =============================================================================
#SBATCH --job-name=simpleavg_rankext_singlejob_improvement_pack_5x20
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# TIME ESTIMATE: Phase 1 (GPU preflight) should add only a few minutes (tiny
# image counts, 1 epoch, 5x20 protocol shape unchanged). Phase 2: see the
# final-readiness report Section 28 -- 9 arms x 6 epochs vs. Exp1's 8 methods
# x 9 epochs (~6h04m observed) scales to ~6h04m x (9x6)/(8x9) ~= 4.55h, plus
# extra SimpleAvg do_merge evaluation passes, serialization, and diagnostics.
# Conservative estimate ~5-5.5h for Phase 2. FINAL-AUDIT FIX: walltime raised
# to the specified 12h (was 08:00:00) -- this is the single available cluster
# GPU allocation and the ~5-5.5h estimate has real variance (queueing/data-
# loading/eval overhead not included in the point estimate); 12h keeps ample
# margin over Phase1+Phase2 combined without meaningfully increasing queueing
# risk on this partition.
#SBATCH --time=12:00:00
#SBATCH --output=simpleavg_rankext_singlejob_improvement_pack_5x20_%j.out
#SBATCH --error=simpleavg_rankext_singlejob_improvement_pack_5x20_%j.err

set -euo pipefail

# FINAL-AUDIT FIX: explicit, absolute repo path -- do NOT derive it from
# `dirname "$0"`. SLURM may execute a COPY of this script from a spool
# directory (e.g. /var/spool/slurm/...), in which case `dirname "$0"` resolves
# to that spool directory, not this repo, and every relative path below it
# would silently point at the wrong (nonexistent, or worse, wrong-version)
# location. `cd` explicitly into the known cluster repo path instead, then
# reference every script by its repo-relative path.
CLUSTER_REPO_DIR="/nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project"
CLUSTER_PYTHON="/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python"
cd "$CLUSTER_REPO_DIR"

echo "================================================================================"
echo "SIMPLEAVG + RANKEXT JOINT SINGLE-JOB IMPROVEMENT PACK -- 5x20, seed 42"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo:   $(pwd)   Python: $CLUSTER_PYTHON"
echo "================================================================================"

# --- Safety: refuse to run if this has been edited into a different config ---
# (the target script's OWN hard assertions -- including the Part-0 readiness
# block that prints+asserts NUM_EPOCHS/arm-counts/rank/schedule/expected-
# configs before any training starts -- are the primary guard; this is a
# fast pre-flight check so a gross misconfiguration is visible in the SLURM
# log header before Python even starts. Epoch checks match the FAST_RUN-
# conditional expressions now in the script, not a bare literal -- the
# script itself still hard-asserts "FAST_RUN or epochs==6" at runtime.)
TARGET_SCRIPT="experiments_prepared/simpleavg_rankext_singlejob_improvement_pack_5x20.py"
if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "FATAL: $TARGET_SCRIPT not found under $CLUSTER_REPO_DIR." >&2
    exit 1
fi
if [ ! -x "$CLUSTER_PYTHON" ]; then
    echo "FATAL: $CLUSTER_PYTHON not found or not executable." >&2
    exit 1
fi
grep -q "^NUM_STEPS = 5$" "$TARGET_SCRIPT" || { echo "FATAL: NUM_STEPS != 5 in $TARGET_SCRIPT" >&2; exit 1; }
grep -q "^CLASSES_PER_STEP = 20$" "$TARGET_SCRIPT" || { echo "FATAL: CLASSES_PER_STEP != 20 in $TARGET_SCRIPT" >&2; exit 1; }
grep -q "LORA_EPOCHS = 1 if FAST_RUN else 6" "$TARGET_SCRIPT" || { echo "FATAL: LORA_EPOCHS real-run value != 6 in $TARGET_SCRIPT" >&2; exit 1; }
grep -q "RANKEXT_EPOCHS = 1 if FAST_RUN else 6" "$TARGET_SCRIPT" || { echo "FATAL: RANKEXT_EPOCHS real-run value != 6 in $TARGET_SCRIPT" >&2; exit 1; }
grep -q "^LORA_R = 80$" "$TARGET_SCRIPT" || { echo "FATAL: LORA_R != 80 in $TARGET_SCRIPT" >&2; exit 1; }
grep -q "^LORA_ALPHA = 2 \* LORA_R$" "$TARGET_SCRIPT" || { echo "FATAL: LORA_ALPHA is not derived as 2*LORA_R (scaling=2 invariant) in $TARGET_SCRIPT" >&2; exit 1; }
grep -q "RANKEXT_RANK_SCHEDULE = \[16, 32, 48, 64, 80\]" "$TARGET_SCRIPT" || { echo "FATAL: RankExt schedule mismatch (Exp2's schedule may be active) in $TARGET_SCRIPT" >&2; exit 1; }
grep -q "^DENSE_ORTH_LAMBDA = 20.0$" "$TARGET_SCRIPT" || { echo "FATAL: DENSE_ORTH_LAMBDA != 20.0 in $TARGET_SCRIPT" >&2; exit 1; }
grep -q '^MERGE_METHODS_TO_EVALUATE = \["arithmetic", "do_merge"\]$' "$TARGET_SCRIPT" || { echo "FATAL: MERGE_METHODS_TO_EVALUATE mismatch in $TARGET_SCRIPT" >&2; exit 1; }
grep -q "^SEED = int(os.environ.get(\"REPLICATION_SEED\", \"42\"))$" "$TARGET_SCRIPT" || { echo "FATAL: SEED default != 42 in $TARGET_SCRIPT" >&2; exit 1; }
echo "Pre-flight file checks PASSED."

unset REPLICATION_SEED || true   # use the file's own default (42), not an inherited override

# Unbuffered Python output, per explicit instruction -- SLURM log lines appear
# as they're printed, not only after the process buffer flushes/exits.
export PYTHONUNBUFFERED=1

# =============================================================================
# PHASE 1 -- GPU PREFLIGHT SMOKE TEST
# =============================================================================
echo ""
echo "================================================================================"
echo "PHASE 1: GPU PREFLIGHT SMOKE TEST (FAST_RUN_DEBUG=1)"
echo "================================================================================"

# Fresh debug output every attempt -- results_debug_preflight/ is NEVER the
# real "results/" tree (a completely separate root, per the script's own
# ROOT_RESULTS_DIR = "results_debug_preflight" if FAST_RUN else "results"),
# so this rm -rf can never touch real scientific output, only ever-disposable
# preflight scratch from a previous attempt.
rm -rf results_debug_preflight
mkdir -p results_debug_preflight

export FAST_RUN_DEBUG=1
set +e
"$CLUSTER_PYTHON" -u "$TARGET_SCRIPT"
PREFLIGHT_EXIT_CODE=$?
set -e
unset FAST_RUN_DEBUG

if [ "$PREFLIGHT_EXIT_CODE" -ne 0 ]; then
    echo "================================================================================"
    echo "===== GPU PREFLIGHT FAILED -- REAL TRAINING ABORTED ====="
    echo "Phase 1 (FAST_RUN_DEBUG) exited with code $PREFLIGHT_EXIT_CODE"
    echo "================================================================================"
    exit 1
fi

# =============================================================================
# PREFLIGHT GATE -- verify structural output, not just exit code
# =============================================================================
PREFLIGHT_CSV="$(ls -t results_debug_preflight/*/tables/simpleavg_singlejob_training_x_merge_results.csv 2>/dev/null | head -1 || true)"
if [ -z "$PREFLIGHT_CSV" ]; then
    echo "================================================================================"
    echo "===== GPU PREFLIGHT FAILED -- REAL TRAINING ABORTED ====="
    echo "Could not find preflight result CSV under results_debug_preflight/"
    echo "================================================================================"
    exit 1
fi
echo "Preflight result CSV: $PREFLIGHT_CSV"

set +e
"$CLUSTER_PYTHON" - "$PREFLIGHT_CSV" <<'PYEOF'
import sys
import pandas as pd

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
errors = []

required_cols = {"family", "training_method", "merge_method", "all_seen"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    errors.append(f"Missing required columns: {sorted(missing_cols)}")
else:
    if len(df) != 14:
        errors.append(f"Expected 14 total rows, got {len(df)}")

    sa = df[df["family"] == "simple_avg"]
    re = df[df["family"] == "rank_extension"]

    if len(sa) != 10:
        errors.append(f"Expected 10 SimpleAvg rows, got {len(sa)}")
    if sa["training_method"].nunique() != 5:
        errors.append(f"Expected 5 distinct SimpleAvg training methods, got {sa['training_method'].nunique()}")
    if set(sa["merge_method"].unique()) != {"arithmetic", "do_merge"}:
        errors.append(f"SimpleAvg merge_method set mismatch: {sorted(sa['merge_method'].unique())}")
    per_arm_ok = sa.groupby("training_method")["merge_method"].apply(lambda s: set(s) == {"arithmetic", "do_merge"})
    if not per_arm_ok.all():
        errors.append(f"Not every SimpleAvg arm has exactly {{arithmetic, do_merge}}: {per_arm_ok[~per_arm_ok].index.tolist()}")

    if len(re) != 4:
        errors.append(f"Expected 4 RankExt rows, got {len(re)}")
    if re["training_method"].nunique() != 4:
        errors.append(f"Expected 4 distinct RankExt training methods, got {re['training_method'].nunique()}")
    if not (re["merge_method"] == "persistent").all():
        errors.append("Not every RankExt row has merge_method == 'persistent'")

    dup_mask = df.duplicated(subset=["training_method", "merge_method"])
    if dup_mask.any():
        errors.append(f"Duplicate (training_method, merge_method) rows: {df.loc[dup_mask, ['training_method', 'merge_method']].to_dict('records')}")

    if df["all_seen"].isna().any():
        missing = df.loc[df["all_seen"].isna(), ["training_method", "merge_method"]].to_dict("records")
        errors.append(f"all_seen has missing/NaN values: {missing}")

if errors:
    print("PREFLIGHT VERIFICATION: FAIL")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("PREFLIGHT VERIFICATION: PASS")
print(f" - {len(df)}/14 rows present")
print(" - 5/5 SimpleAvg training paths x {arithmetic, do_merge} = 10/10 evaluations")
print(" - 4/4 RankExt training paths x persistent = 4/4 evaluations")
print(" - no duplicate (training_method, merge_method) rows")
print(" - all_seen fully populated (no NaN) for all 14 rows")
sys.exit(0)
PYEOF
VERIFY_EXIT_CODE=$?
set -e

if [ "$VERIFY_EXIT_CODE" -ne 0 ]; then
    echo "================================================================================"
    echo "===== GPU PREFLIGHT FAILED -- REAL TRAINING ABORTED ====="
    echo "================================================================================"
    exit 1
fi

echo "================================================================================"
echo "===== GPU PREFLIGHT PASS -- STARTING REAL EXPERIMENT ====="
echo "================================================================================"

# =============================================================================
# PHASE 2 -- REAL EXPERIMENT (FAST_RUN_DEBUG unset/off)
# =============================================================================
echo ""
echo "================================================================================"
echo "PHASE 2: REAL EXPERIMENT -- 9 training arms, 14 evaluated configurations, 6 epochs"
echo "Start: $(date -Is)"
echo "================================================================================"

echo "Launching: $CLUSTER_PYTHON -u $TARGET_SCRIPT"
"$CLUSTER_PYTHON" -u "$TARGET_SCRIPT"

echo "================================================================================"
echo "SIMPLEAVG + RANKEXT JOINT SINGLE-JOB IMPROVEMENT PACK finished. End: $(date -Is)"
echo "================================================================================"
