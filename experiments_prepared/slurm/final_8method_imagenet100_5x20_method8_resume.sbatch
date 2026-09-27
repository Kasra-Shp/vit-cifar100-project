#!/bin/bash
# Fail-closed Method-8-only recovery.  Tasks 1-3 are loaded from the exact
# atomic boundary checkpoint; Task 4 restarts at epoch 0, then Task 5 runs.
# No methods 1-7 are enabled or trained by this launcher.
#SBATCH --job-name=imagenet100_method8_resume_ep9
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --output=imagenet100_method8_resume_%j.out
#SBATCH --error=imagenet100_method8_resume_%j.err

set -euo pipefail

PROJECT_ROOT=/nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8method_imagenet100_5x20_method8_resume.py
VERIFIER=tools/verify_imagenet100_dataset.py
export IMAGENET100_ROOT="${IMAGENET100_ROOT:-/nfsd/lttm4/datasets/ImageNet-1k_torch}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
[ -x "$PYTHON" ]
[ -f "$SCRIPT" ]
[ -f "$VERIFIER" ]
[ -d "$IMAGENET100_ROOT/train" ]
[ -d "$IMAGENET100_ROOT/val" ]

RUNTIME_MANIFEST=results/imagenet100_5x20_final_8method_canonical_continuation_methods6to8_seed42_ep9_EPOCH9_MAIN_20260927_012329/configs/protocol_manifest.json
VALIDATION_OUTPUT=$("$PYTHON" tools/validate_imagenet100_method8_resume.py \
  --checkpoint "results/imagenet100_5x20_final_8method_canonical_continuation_methods6to8_seed42_ep9_checkpoints/rank_extension_factor_orth_lam50_fullkd_T2_protect30__rankext_state.pt" \
  --runtime-manifest "$RUNTIME_MANIFEST")
printf '%s\n' "$VALIDATION_OUTPUT"
printf '%s\n' "$VALIDATION_OUTPUT" | grep -Fqx "CHECKPOINT VALID: YES"
echo "Method-8 checkpoint validation passed; training may proceed."
"$PYTHON" "$VERIFIER" --data-root "$IMAGENET100_ROOT" --source-mode shared_imagenet1k --max-readable-checks 24
unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true
"$PYTHON" -u "$SCRIPT"
