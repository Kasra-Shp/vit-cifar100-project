# Prepared, NOT launched: supervisor CIFAR-100 followup experiments

See `thesis_agent/reports/supervisor_cifar_followup_audit.md` for the full audit that produced
this directory. **Nothing here has been submitted.** Do not `sbatch` anything in `slurm/` without
explicit author review.

| File | Experiment |
|---|---|
| `supervisor_exp1_cifar100_5x20_fixed_rankext.py` | Experiment 1 — 5x20 re-evaluation under the current fair/canonical pipeline, corrected RankExt implementation, historical RankExt schedule (`[16,32,48,64,80]`). Renamed 2026-09-11 from `vit_lora_cifar100_5x20_fixed_rankext_seed42_EXP1.py`. |
| `vit_lora_cifar100_5x20_rank32_capacity_seed42_EXP2.py` | Experiment 2 — 5x20, SimpleAvg rank 32 (alpha 64, scaling 2.0) vs. RankExt `[32,64,96,128,160]` (scaling 2.0), the supervisor's requested deliberate-capacity-advantage compromise |
| `slurm/exp1_5x20_fixed_rankext_seed42.sbatch` | Launcher for Experiment 1 (`gpu:a40:1`, `allgroups`, not submitted). **STALE PATH (2026-09-11):** still targets the pre-rename filename `vit_lora_cifar100_5x20_fixed_rankext_seed42_EXP1.py` — update its `TARGET_SCRIPT` line to `supervisor_exp1_cifar100_5x20_fixed_rankext.py` before use. Left untouched deliberately (a request explicitly excluded editing SLURM files from that pass). |
| `slurm/exp2_5x20_rank32_capacity_seed42.sbatch` | Launcher for Experiment 2 (`gpu:a40:1`, `allgroups`, not submitted) |

Both `.py` files are dedicated copies of the restored canonical `vit_lora_cifar100_full5step_n5.py`
(pre-ImageNet, seed-42-default, fixed-RankExt state) with **only** the protocol/rank/alpha/epoch
constants documented in each file's own header comment changed — no SimpleAvg/RankExt/KD/
FactorOrth/calibration/metric algorithm was touched. Each file carries its own hard, fail-fast
assertions (dataset/protocol/seed/method-set/rank/alpha/scaling/KD/FactorOrth) and prints a
startup diagnostic block before any training begins.

If canonical `vit_lora_cifar100_full5step_n5.py` is ever changed again (a new fix, a new metric),
re-derive these two files from the new canonical version rather than hand-patching the copies here,
to avoid silently drifting from a fix these experiments are specifically meant to test.
