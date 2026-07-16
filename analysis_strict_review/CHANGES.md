# CHANGES.md — code edits for the next instrumented run

All edits are in `vit_lora_cifar100_full5step_n5.py`. No training was run this
session; every change below is verified by `py_compile` + `pyflakes` (both
clean) and, where possible, dry-run against saved CSVs or synthetic data (see
`report.txt` for the numeric evidence behind each decision). Search the file
for `STRICT-REVIEW` to find every marked block.

## A1 — Combined-loss-decomposition figure redesign (~line 6001)
Replaced the 4-rows(loss component) x 2-cols(family) layout — which put a
variant's CE (linear row) and its Total (log row) on two different panels
with two different scales and no legend — with a **1-panel-per-variant**
layout (4 rows = variant, 2 cols = family). Every panel now plots Train CE,
KD weighted, Factor-Orth weighted, and TOTAL together on **one shared log
axis**, plus a single shared legend (`fig.legend`) that was completely absent
before. This makes "TOTAL = sum of the other three" checkable by eye within
one panel instead of inferred by comparing two panels with different scales.
Purely a plotting change; no effect on any logged metric or training
behavior.

## B1 — Merge-mechanism logging for simple_avg (~line 2187, hook at ~line 3221)
Added `log_merge_mechanism()` (a pure diagnostic function, right after
`simple_average_deltas()`) and one call site inside `run_simple_avg_variant()`
immediately after `merged_delta = simple_average_deltas(step_states)`. For
every target module, logs (a) each task's pre-merge `||dW_t||` and
`cos(dW_1, dW_t)`, and (b) the merged delta's post-merge norm and
`cos(dW_1, merged)`, to `tables/merge_mechanism_by_method_step.csv`. Runs for
all 4 simple_avg variants (their shared code path). Operates only on
already-extracted CPU tensors already computed regardless of this change
(`extract_lora_state()` output, `merged_delta`); no forward/backward pass, no
new model, no change to `merged_delta` or anything downstream. Dry-run
validated against synthetic healthy/destructive-dilution matrices (see
`dryrun_merge_mechanism_logging.py` / `dryrun_merge_mechanism.csv`) — the
arithmetic matches theoretical predictions (healthy case ≈ 1/√5 variance
reduction from averaging 5 independent random deltas) and correctly flags an
engineered destructive-cancellation case (`cos(dW1, merged)` ≈ -0.997).

## B2 — Combined-variant second iteration (~line 410, ~line 2927)
- `COMBINED_ORTH_WARMUP_ENABLED`: `False` → `True` (was already fully
  implemented and gated correctly; just flipped the flag).
- `COMBINED_LAMBDA_ORTH_SCALE`: `0.5` → `0.3` (lambda_orth 25 → 15 for
  `simple_avg_factor_orth_kd_T2` only).
- `COMBINED_KD_WEIGHT_SCALE`: **left at `0.5`** (NOT raised to ~0.7 as
  originally floated) — the NEW run's steady-state kd/ce ratio for this
  method (1.873) already tracks `simple_avg_kd_T2`'s own natural,
  full-strength ratio (1.757) closely; there is no evidence KD is
  under-weighted, and later_steps (new-class accuracy) is already this
  method's weakest metric relative to its single-mechanism siblings, so
  raising kd_weight further has no supporting evidence and a plausible
  downside. Documented in-line with the exact numbers.
- Updated two now-stale "off (the default)" doc comments near the trainer's
  per-batch logging dict (lines ~2927, ~2965) to reflect the new default.

## B3 — rank_extension: enable ONE lever (~line 654)
- `RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED`: `False` → `True`.
- `USE_RANKEXT_RANK_SCHEDULE_WIDE`: **left at `False`** (confirmed untouched)
  — single-lever discipline for clean attribution in the next rerun.
- Re-verified the justifying evidence directly against the NEW/revert run
  (not just the earlier calibfix-run numbers already in the comment):
  train_ce_loss ratio (orth vs plain, local_epoch==1) grows 1.64x→1.89x→2.31x→
  2.04x across steps 2-5, and factor_orth_loss_weighted collapses 99.7-99.8%
  within one epoch at every step transition — confirms the spike is real,
  persists in the reverted baseline, and is short enough for a 1-epoch warmup
  to skip past.
- Updated one stale "off (the default)" doc comment near the rank_extension
  trainer's per-batch logging dict (line ~4346).

## B4 — Seed readiness (~line 100, ~5754, ~5410-5446, ~5792-5867)
- `SEED` (line 100) was already the single constant driving every source of
  randomness (`set_seed`/`random.seed`/`np.random.seed`/`torch.manual_seed`,
  plus every per-class/per-step dataset shuffle via `SEED + offset`) — no
  second hardcoded seed existed anywhere else in the file (verified by
  grepping for `seed=` and `_seed`). Expanded its comment to state this
  explicitly and point at every place it now gets logged.
- Added a `"seed"` column, propagated into:
  - `cfg_df()` (→ `configs/hyperparameters_by_method.json`)
  - `tables/hyperparameter_consistency_check.csv`
  - `tables/method_hyperparameter_summary.csv`
  - `tables/supervisor_selected_accuracy_comparison.csv`
  - `tables/final_metrics_all_methods.csv`
  - `tables/training_loss_history_by_epoch.csv`
  - (`configs/run_config.json` already had `"seed":SEED` from before this
    session — unchanged.)
- A multi-seed sweep now requires changing only the `SEED = 42` line; every
  saved table/JSON is self-describing about which seed produced it.

## Verification performed this session
- `python -m py_compile vit_lora_cifar100_full5step_n5.py` — clean, after
  every edit above.
- `python -m pyflakes vit_lora_cifar100_full5step_n5.py` — clean (exit 0, no
  warnings).
- `log_merge_mechanism()` dry-run on synthetic matrices (see B1 above).
- A1's redesigned figure and A3's documentation figure both regenerated from
  the NEW run's real saved CSVs (no training) to confirm the plotting code
  runs correctly end-to-end — see `regenerate_combined_loss_decomposition.py`
  and `make_A3_kd_rise_figure.py`.
- `print_final_config_table.py` parses the actual post-edit constant values
  back out of the source file (not hand-transcribed) and reconstructs the
  full 8-method config table exactly as `cfg_df()` would compute it, to
  confirm every flag/value lands where intended (see
  `final_per_method_config_table_for_next_run.csv`).
