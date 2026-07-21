# CHANGES.md — code edits for the next cluster run (Fix 2)

All edits are in `vit_lora_cifar100_full5step_n5.py`. No training was run
this session; verified by `py_compile` + `pyflakes` (both clean) and a
proxy dry-run against the Fix 1 run's saved diagnostics (see `report.txt`
Task D.3 — Fix 1's run has no saved classifier weight tensors, only shapes,
so the literal function could not be re-invoked on real tensors; the dry-run
instead reproduces the new function's exact arithmetic on Fix 1's saved
pre-calibration row norms and per-step val_ce_loss).

## 1 — Confidence-weighted regime-grouped classifier calibration for
rank_extension (FIX 2)

**Why:** `report.txt` Task A shows Fix 1 (regime-grouped mean-norm
calibration) achieved essentially perfect row-norm equalization for the
non-KD `rank_extension` variants (ratio 1.0000000 across all 5 steps,
`tables/classifier_row_norm_diagnostics_by_method_step.csv` in the Fix 1
run) while open accuracy for early/middle steps barely moved (single-digit-pp
changes against a 55-95pp restricted-accuracy ceiling) — direct proof that
mean row-norm imbalance was never the dominant driver of the open-vs-
restricted gap. `report.txt` Task B instead finds the gap correlates
strongly (pearson +0.85, +0.91) with each step's own final-epoch validation
CE loss, specifically within the group Fix 1's KD-variant calibration
actually touches (steps 2-5) — NOT with row-norm ratio, which is confounded
with step position and has the WRONG sign. Fix 2 redirects the same
scale-correction mechanism at that demonstrated correlate instead of the
falsified one.

- **`RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED`, `line 381`**
  (new flag): `True`. Layered on top of the existing
  `RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED` (Fix 1's master switch,
  unchanged, still `True`) — Fix 2 only activates when both flags are `True`.
- **`CALIBRATION_MODE_BY_FAMILY`, `~line 402-419`**: `"rank_extension"` now
  resolves to `"confidence_weighted_regime_grouped"` when both flags are
  `True`, `"regime_grouped"` (Fix 1, unchanged) when only
  `RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED` is `True`, `"off"` otherwise.
  `"simple_avg"` is unchanged (`"global"`, always) — its call site in
  `run_simple_avg_variant()` (`~line 3746`) can never reach the new mode or
  the new function.
- **`classifier_confidence_calibration_diagnostic_rows`, `line 1143`**
  (new module-global list): accumulates one row per (method, step_id)
  actually processed by the new function — the val_ce_loss used, the group's
  mean val_ce_loss, the resulting `relative_difficulty` and `boost_factor`,
  and the pre/post target row norms. Written to
  `tables/classifier_confidence_calibration_diagnostics_by_method_step.csv`
  at the end of the run (see item below) — lets the next run's report show
  the boost factors that were actually applied, not just infer them.
- **`calibrate_classifier_row_norms_confidence_weighted()`, new function,
  `line 2894-3036`** (added immediately after
  `calibrate_classifier_row_norms()`, which is left byte-for-byte
  unchanged — per the task's requirement not to patch it in place, so
  `mode="global"` and `mode="regime_grouped"` both remain fully runnable and
  directly comparable to Fix 1's own results):
  - Takes `model, epoch_loss_rows, method_name, eps=1e-8, uses_kd=False,
    gamma=0.5, boost_min=0.85, boost_max=1.3`.
  - Uses the IDENTICAL step grouping as
    `calibrate_classifier_row_norms(mode="regime_grouped")`: KD →
    `{step1}` singleton no-op + `{steps2..N}` one group; non-KD → one group
    covering every step. Grouping is duplicated (not shared via a helper) to
    avoid any risk of the new function's behavior drifting the existing,
    already-tested function.
  - Reads each step's own FINAL-EPOCH validation CE loss from the
    module-global `epoch_loss_rows` accumulator (already populated by
    `EpochValidationCallback` during training — no new training-time
    instrumentation was needed; this data already exists by the time
    calibration runs, at the end of `run_rank_extension_variant()`).
  - Per non-singleton group: `group_target_norm` is the plain mean of the
    group's pre-calibration row norms (identical formula to Fix 1). Per step
    in the group: if `uses_kd` is `True`, `relative_difficulty =
    step_val_ce / group_mean_val_ce`, `boost = clamp(relative_difficulty **
    gamma, boost_min, boost_max)`, `target_norm = group_target_norm *
    boost`. If `uses_kd` is `False`, `boost` is unconditionally `1.0` for
    every step — this makes the function numerically IDENTICAL to
    `calibrate_classifier_row_norms(mode="regime_grouped")` for the two
    non-KD `rank_extension` variants, by construction (see `report.txt`
    Task B.c: val_ce_loss does not track the gap for the non-KD variants,
    so boosting on it there would be acting on noise, with a real risk of
    active harm rather than just no benefit).
  - Logs pre/post row-norm diagnostics via the EXISTING
    `log_classifier_row_norm_diagnostics()` (unchanged — same table/columns
    as Fix 1, full 8-method coverage preserved) plus a new diagnostic row
    per (method, step_id) into `classifier_confidence_calibration_
    diagnostic_rows` (see above).
  - Only rescales `model.classifier.weight` rows, not the bias — same
    convention as `calibrate_classifier_row_norms()`.
- **Call site — `run_rank_extension_variant()`, `~line 5245-5266`**: branches
  on `ACTIVE_METHOD_MAP[method_name]["calibration_mode"]`. When it equals
  `"confidence_weighted_regime_grouped"`, calls the new function (passing
  the module-global `epoch_loss_rows`); otherwise falls through to the
  existing `calibrate_classifier_row_norms(mode=calibration_mode, ...)` call,
  unchanged. This is the ONLY call site that can ever reach the new
  function — `run_simple_avg_variant()`'s calibration call
  (`~line 3746-3753`) is untouched, since `simple_avg`'s `calibration_mode`
  is always `"global"`.
- **Startup print block, `line 1387` and `line 1662`**: added
  `RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED` to the existing
  `print("\nRank extension:")` dict and as a new print line, matching how
  every other flag in this file is surfaced at run start.
- **`run_config.json`, `line 6366` (single dict literal)**: added
  `"rankext_confidence_weighted_calibration_enabled":
  bool(RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED)` next to the
  existing `"rankext_family_aware_calibration_enabled"` key.
  `hyperparameters_by_method.json`'s `"calibration_mode"` column needs no
  new wiring — it already reads `family_calibration_mode(family)` inside
  `add_method()`, which automatically picks up the new mode string once
  `CALIBRATION_MODE_BY_FAMILY` resolves to it.
- **Diagnostics CSV write, `~line 6446-6459`** (added immediately after the
  existing row-norm diagnostics CSV write): builds and writes
  `classifier_confidence_calib_diag_df` from
  `classifier_confidence_calibration_diagnostic_rows` to
  `tables/classifier_confidence_calibration_diagnostics_by_method_step.csv`,
  same empty-DataFrame-with-columns fallback convention as the row-norm
  diagnostics table above it.

**Risk / protocol note:** still a pure post-hoc, per-step-UNIFORM rescale of
the already-merged classifier (no retraining, no new parameters, no
rehearsal data, restricted accuracy remains exactly invariant by
construction) — same category of change as Fix 1 and the pre-existing
`simple_avg` calibration, not a new mechanism. The `uses_kd` gate makes Fix 2
provably no worse than Fix 1 for the two non-KD `rank_extension` variants (by
construction, not by tuning); the new, less-validated boost logic is confined
to exactly the two variants and exactly the group (steps 2-5) where
`report.txt` Task B's evidence supports it. Step 1's no-op status (for the
KD variants) is unchanged from Fix 1 — `report.txt` Task B.a/B.b found no
evidence it was hurting step 1 relative to the calibrated steps, so the
task's "revert if hurting step 1" condition was checked and not triggered.

## Verification

- `python -m py_compile vit_lora_cifar100_full5step_n5.py` → clean (no
  output, exit 0).
- `python -m pyflakes vit_lora_cifar100_full5step_n5.py` → clean (zero
  warnings).
- `final_per_method_config_table.py` / `.csv` (this directory): standalone
  reproduction of `add_method()`'s resolution logic (no torch/dataset
  dependency) — confirms all 4 `simple_avg` methods are byte-identical to
  before (`calibration_mode="global"`); all 4 `rank_extension` methods now
  resolve to `calibration_mode="confidence_weighted_regime_grouped"`, with
  an `effective_calibration_behavior` column confirming the `uses_kd` gate
  gives the two non-KD variants exactly Fix 1's old behavior and the two KD
  variants the new boost logic.
- `dry_run_row_norm_projection.py` / `.csv` (this directory): reproduces the
  new function's exact arithmetic against the Fix 1 run's saved
  pre-calibration row norms and per-step val_ce_loss (Fix 1's run has no
  saved classifier weight tensors to literally replay the fix against, same
  constraint noted in `analysis_recency_fix/report.txt` A.2). Confirms boost
  factors are exactly 1.0 for every non-KD step (no behavior change) and
  meaningfully differentiated (0.90-1.11x) for the KD variants' steps 2-5,
  concentrated at step 2 (the worst-affected step per `report.txt` Task B).
- `report.txt` Task D.4: grounded, non-optimistic projection (single-digit
  pp on the worst-affected steps of the FactorOrth+KD variant specifically;
  ~0 change for the non-KD variants and for `rank_extension_kd_only_T2`'s
  smaller/noisier signal) — explicitly does NOT repeat
  `analysis_recency_fix/projected_improvement.csv`'s +12 to +57pp range,
  which `report.txt` Task A.6 traces to an assumption (row-norm imbalance is
  the dominant driver) the actual Fix 1 result falsified.
