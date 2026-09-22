#!/bin/bash
# CUB-200-2011 full 200-class, 5x40 job4971615-compatible 8-method launcher, epochs=5.
# Created for later submission; this file is intentionally not submitted here.
# It points only to the new CUB script and never mutates canonical CIFAR-100 or
# ImageNet experiment files.
#SBATCH --job-name=final_8method_cub200_5x40_job4971615_ep5
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=final_8method_cub200_5x40_job4971615_ep5_%j.out
#SBATCH --error=final_8method_cub200_5x40_job4971615_ep5_%j.err

set -euo pipefail

cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8method_cub200_5x40_job4971615_ep5.py
ARTIFACT=experiments_prepared/splits/cub200_class_order_seed42_5x40.json
RUN_NAME_BASE=cub200_5x40_final_8method_job4971615_compatible_seed42_ep5

# Shared cluster caches; no home-directory cache is used by this launcher.
SHARED_ROOT=/nfsd/lttm4/tesisti/shahrampour
export CUB200_CACHE_ROOT="${CUB200_CACHE_ROOT:-$SHARED_ROOT/datasets/huggingface}"
export HF_HOME="${HF_HOME:-$SHARED_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SHARED_ROOT/.cache/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SHARED_ROOT/.cache/huggingface/datasets}"

fatal() { echo "FATAL: $*" >&2; exit 1; }

[ -x "$PYTHON" ] || fatal "interpreter missing or not executable: $PYTHON"
[ -f "$SCRIPT" ] || fatal "CUB experiment script missing: $SCRIPT"
[ -f "$ARTIFACT" ] || fatal "persistent class-order artifact missing: $ARTIFACT"
[ "$(basename "$SCRIPT")" = "final_8method_cub200_5x40_job4971615_ep5.py" ] || fatal "launcher does not point to the required CUB script"

mkdir -p "$CUB200_CACHE_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
[ -d "$CUB200_CACHE_ROOT" ] || fatal "CUB cache directory could not be created: $CUB200_CACHE_ROOT"
[ -w "$CUB200_CACHE_ROOT" ] || fatal "CUB cache directory is not writable: $CUB200_CACHE_ROOT"
[ "$CUB200_CACHE_ROOT" != "$HOME/.cache/huggingface/datasets" ] || fatal "home-directory dataset cache is forbidden"

"$PYTHON" -c 'import sys, datasets, transformers, torch; print("interpreter:", sys.executable); print("datasets:", datasets.__version__); print("transformers:", transformers.__version__); print("torch:", torch.__version__)'
"$PYTHON" -c 'import json, sys; p=sys.argv[1]; d=json.load(open(p, encoding="utf-8")); assert d["dataset_identifier"] == "Donghyun99/CUB-200-2011"; assert d["seed"] == 42; assert d["num_classes"] == 200; assert d["num_tasks"] == 5; assert d["classes_per_task"] == 40; assert len(d["original_id_to_new_id"]) == 200; assert len(d["new_id_to_original_id"]) == 200; assert len(d["original_class_names_by_original_id"]) == 200; assert len(d["tasks"]) == 5; assert all(len(t["new_class_ids"]) == 40 for t in d["tasks"]); print("class-order artifact: PASS")' "$ARTIFACT"

unset FAST_RUN_DEBUG || true
unset CUB200_PREFLIGHT_ONLY || true
unset REPLICATION_SEED || true
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo "CUB-200-2011 JOB4971615-COMPATIBLE 8-METHOD RUN (EPOCHS=5)"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo: $PWD"
echo "Python: $PYTHON"
echo "Script: $SCRIPT"
echo "RUN_NAME_BASE: $RUN_NAME_BASE"
echo "CUB dataset: Donghyun99/CUB-200-2011@main/default (official train/test split)"
echo "Protocol: full 200 classes, 5 tasks x 40 classes, seed 42"
echo "CUB cache: $CUB200_CACHE_ROOT"
echo "Shared HF home: $HF_HOME"
echo "NOTE: if the dataset/model is not already cached, the compute node must have"
echo "outbound access to huggingface.co and its CDN; this launcher does not submit"
echo "or pre-download data on its own."
echo "================================================================================"

"$PYTHON" -u "$SCRIPT"

echo "================================================================================"
echo "CUB-200-2011 job4971615-compatible run finished. End: $(date -Is)"
echo "================================================================================"
