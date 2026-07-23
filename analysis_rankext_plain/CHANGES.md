# CHANGES.md -- analysis_rankext_plain (2026-07-23)

All edits are in `vit_lora_cifar100_full5step_n5.py`. No training was run this
session; these changes take effect on the next cluster run. EPOCHS (9),
LORA_DROPOUT, every calibration flag, all simple_avg config, the merging
code, KD temperature, factor-orth lambda, and SEED were all left untouched,
per this session's pre-authorizations.

## Summary

Implements **Candidate 2** from `candidate_evaluation.txt`: a within-step
warmup on the newly-added LoRA rank block's *output contribution* to the
forward pass (not its weights, not the frozen old block, not the classifier
head), ramping linearly from 0 to full strength over the first
`RANKEXT_NEW_BLOCK_WARMUP_EPOCHS` (= 1.0) epochs of each CL step's local
training. Applied identically to all four active rank_extension variants
(plain, +FactorOrth, +KD, +FactorOrth+KD); never applied to simple_avg.

## New flags

- `RANKEXT_NEW_BLOCK_WARMUP_ENABLED = True` (default ON, per
  pre-authorization #3). Existing (no-warmup, full strength from batch 1)
  behavior is exactly preserved when `False` -- verified: with
  `enabled=False`, `orth_lambda_warmup_multiplier()` always returns `1.0`
  regardless of epoch, so `GrowingRankLoRALinear.forward()`'s new-block
  branch is numerically identical to its pre-this-session form.
- `RANKEXT_NEW_BLOCK_WARMUP_EPOCHS = 1.0` -- deliberately reuses the exact
  value already validated for the structurally identical
  `RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS`, per pre-authorization #5 ("pick the
  more conservative value" -- reusing an already-proven duration rather than
  guessing a new one is the conservative choice here).

Both placed near `RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED` (its existing
sibling mechanism), with a long rationale comment tying the fix back to
`analysis_rankext_plain/diagnosis.txt`.

## New module-level state

- `_rankext_new_block_warmup_state = {"multiplier": 1.0}` + getters/setters
  `set_rankext_new_block_warmup_multiplier()` /
  `get_rankext_new_block_warmup_multiplier()`. Always `1.0` outside the
  narrow window of an active rank_extension training step -- guaranteed by
  an unconditional reset both immediately before and immediately after every
  `trainer.train()` call inside `train_with_trainer()` (belt-and-suspenders,
  same redundancy pattern this codebase already uses for classifier-row
  restore).
- `family_uses_new_block_warmup(family)` -- `True` only for
  `family == "rank_extension"` (and only when the flag is on). simple_avg is
  excluded both by this gate AND structurally: `GrowingRankLoRALinear` is the
  only module class that ever reads the multiplier; simple_avg's PEFT LoRA
  layers do not.
- `rankext_new_block_warmup_diagnostic_rows = []` -- module-level
  accumulator, one row per (method, step, local_epoch) actually trained,
  populated by the new callback below.

## `GrowingRankLoRALinear.forward()`

The new-block branch now reads:
```python
new_block_multiplier = get_rankext_new_block_warmup_multiplier()
out = out + self.scaling * lora_new * new_block_multiplier
```
Only this branch is touched. The frozen-old-block branch (`old_active_in_
forward and frozen_rank > 0`) and `base_out` are completely unaffected --
this fix cannot touch the frozen mechanism diagnosis.txt confirmed clean.

## New callback: `RankExtNewBlockWarmupCallback`

Placed next to `ClassifierRowRestoreCallback`. `on_step_begin` updates the
module-level multiplier from `state.epoch` every step (so the ramp is smooth
within an epoch, not just stepped at epoch boundaries); `on_epoch_begin`
logs one diagnostic row per local epoch (not per batch, to keep the saved
table small); `on_train_end` resets the multiplier to 1.0 as a second,
redundant safety net on top of `train_with_trainer()`'s own reset.

## `train_with_trainer()`

Two new optional parameters, both defaulting to `None`:
- `rankext_new_block_warmup_epochs=None`
- `rankext_new_block_warmup_diagnostic_records=None`

Neither is passed by simple_avg's call site (`train_independent_loras()`),
so simple_avg's training is provably unaffected by this change -- not merely
gated off, but never invoked. Behavior:
- Unconditionally resets the multiplier to `1.0` both before and after
  `trainer.train()`, regardless of family or flag state.
- If `rankext_new_block_warmup_epochs is not None`, attaches
  `RankExtNewBlockWarmupCallback` before calling `trainer.train()`.

## `run_rank_extension_variant()`

- New local: `active_new_block_warmup_epochs = float(RANKEXT_NEW_BLOCK_
  WARMUP_EPOCHS) if family_uses_new_block_warmup("rank_extension") else
  None` -- resolved once per method (not per step), since this function is
  only ever called for rank_extension-family methods, so all 4 active
  variants resolve identically (same protocol per family, per
  pre-authorization #2).
- The `train_with_trainer(...)` call now passes
  `rankext_new_block_warmup_epochs=active_new_block_warmup_epochs` and
  `rankext_new_block_warmup_diagnostic_records=rankext_new_block_warmup_
  diagnostic_rows`.

## `build_active_method_configs()` / `add_method()`

Two new per-method fields recorded in every method's config dict (flows
automatically into `ACTIVE_METHOD_MAP`, `method_config_df`, and
`hyperparameters_by_method.json` via the existing `CFG.to_dict("records")`
write, no new plumbing needed):
- `"rankext_new_block_warmup_enabled": family_uses_new_block_warmup(family)`
- `"rankext_new_block_warmup_epochs": float(RANKEXT_NEW_BLOCK_WARMUP_EPOCHS)
  if family_uses_new_block_warmup(family) else 0.0`

## `run_config.json`

Two new top-level keys added to the existing `js(Path(CONFIGS_DIR)/
"run_config.json", {...})` call:
- `"rankext_new_block_warmup_enabled": bool(RANKEXT_NEW_BLOCK_WARMUP_ENABLED)`
- `"rankext_new_block_warmup_epochs": float(RANKEXT_NEW_BLOCK_WARMUP_EPOCHS)`

## New diagnostic table

`tables/rankext_new_block_warmup_diagnostics_by_method_step_epoch.csv` --
one row per (method, step, local_epoch) actually trained, columns:
`method_name, step_id, local_epoch, new_block_warmup_multiplier,
warmup_epochs_configured`. Empty (header-only) when the flag is off, since
the populating callback is never attached in that case. Written near the
existing `best_epoch_selected_by_method_step.csv` /
`growing_overfitting_diagnostics_by_method_step.csv` writes, following the
same pattern.

## What was explicitly NOT touched

EPOCHS, LORA_DROPOUT, RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED,
RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED, CALIBRATION_MODE_BY_FAMILY,
any simple_avg config or code path, `calibrate_classifier_row_norms*()`, the
merging/`apply_deltas_to_base()` code, `KD_TEMPERATURES`/`kd_weight`,
`LAMBDA_ORTH`, `SEED`. `calibrate_classifier_row_norms()` and `calibrate_
classifier_row_norms_confidence_weighted()` are byte-for-byte unchanged.
Candidates 1, 3, 4, 5 from `candidate_evaluation.txt` were NOT implemented
(candidate 4 because no drift was found to harden; 1/3/5 because candidate 2
ranked highest -- see that file for the full comparison).

## Verification performed this session

- `python -m py_compile vit_lora_cifar100_full5step_n5.py` -- PASS
- `python -m pyflakes vit_lora_cifar100_full5step_n5.py` -- PASS (0 warnings)
- Standalone dry-run of `orth_lambda_warmup_multiplier()` (the reused ramp
  formula) confirms: linear 0->1 ramp over epoch in [0,1], holds at 1.0 for
  epochs 1-9, and returns exactly 1.0 for every epoch when `enabled=False` --
  i.e. the flag-off path is numerically proven identical to pre-existing
  behavior, not just intended to be.
- `final_per_method_config_table.csv/.py` (this directory) confirms EPOCHS=9,
  the new flag's per-family resolution, and every other unchanged flag's
  value, via a standalone (no torch/GPU) reproduction of `add_method()`'s
  resolution logic.
