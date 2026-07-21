# CHANGES.md — code edits for the next cluster run

All edits are in `vit_lora_cifar100_full5step_n5.py`. No training was run
this session; verified by `py_compile` + `pyflakes` (both clean) and a
proxy dry-run against R5's saved accuracy CSVs (see `report.txt` Task B.5 —
R5 has no saved logits/weight values, only shapes, so the literal fix
function could not be re-invoked on real tensors; the dry-run instead uses
the scale-invariance identity documented in `report.txt` B.0).

## 1 — Family-aware, regime-grouped classifier calibration for rank_extension (FIX 1, chosen fix)

**Why:** `report.txt` Task A traces R5's low OPEN-argmax numbers for all four
`rank_extension` variants to classifier recency bias, not real forgetting —
restricted (step-local) accuracy is 80-97% at every step while open accuracy
for early/middle steps collapses. `calibrate_classifier_row_norms()` already
existed and already fixes exactly this for `simple_avg`, but was fully
disabled for `rank_extension` (`CALIBRATION_ENABLED_FAMILIES["rank_extension"]
= False`) after an earlier incident where applying its single GLOBAL target
norm corrupted `rank_extension`'s two KD variants (68.0%→26.2%, 59.3%→50.1%)
by mixing step 1's untouched, teacher-less rows into the same target as
steps 2-N's KD-trained rows. See `report.txt` Task B for the three-fix
comparison and why this one was chosen over task-ID-restricted argmax and
logit-norm equalization.

- **`CALIBRATION_ENABLED_FAMILIES`, `~line 302-349`**: new flag
  `RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED = True` added just above this
  dict; `CALIBRATION_ENABLED_FAMILIES["rank_extension"]` changed from the
  literal `False` to `bool(RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED)` (so the
  new flag is the single source of truth, and reverting is one line).
- **`CALIBRATION_MODE_BY_FAMILY`, `~line 351-361`** (new): per-family
  calibration *algorithm* selector — `"simple_avg": "global"` (unchanged
  behavior), `"rank_extension": "regime_grouped" if
  RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED else "off"`.
- **`family_calibration_mode(family)`, `~line 371-375`** (new): accessor for
  the dict above, mirrors the existing `family_applies_calibration()` /
  `family_target_modules()` pattern.
- **`add_method()`, `~line 908`**: per-method config dict now includes
  `"calibration_mode"` (via `family_calibration_mode(family)` when
  calibration applies, else `"off"`) — flows through to
  `hyperparameters_by_method.json` automatically (it's just a column of
  `ACTIVE_METHOD_CONFIGS`, no separate wiring needed) and into
  `run_config.json` via a new top-level key (see item 3 below).
- **`log_classifier_row_norm_diagnostics()`, `~line 2707-2732`** (new):
  read-only diagnostic — appends one row per CL step to a new module-global
  `classifier_row_norm_diagnostic_rows` accumulator (declared `~line 1074`,
  next to `per_step_accuracy_restricted_rows`), recording that step's
  classifier weight-row-block mean norm and its ratio to step 1's mean norm.
  Written to `tables/classifier_row_norm_diagnostics_by_method_step.csv` at
  the end of the run (`~line 6206-6213`) — the direct numerical answer to
  Task A.2 ("are later steps' rows several times larger?"), which R5 could
  not answer because no prior version of this script ever logged classifier
  row values (only shapes, via the pre-existing `*_trainable_parameters.csv`
  tables).
- **`calibrate_classifier_row_norms()`, `~line 2738-2823`** (rewritten):
  added `mode="global"` (default, byte-identical behavior to the old
  function), `uses_kd=False`, and `method_name=None` parameters.
  `mode="regime_grouped"` (only meaningfully different when `uses_kd=True`)
  partitions the `NUM_STEPS` step-blocks into `[[0], [1..NUM_STEPS-1]]`
  instead of one `[0..NUM_STEPS-1]` group, and computes an independent
  target norm per group instead of one global target — group `[0]` (step 1)
  is a singleton, so its "target" equals its own current mean, i.e. a
  deliberate no-op that never touches step 1's rows using KD-regime
  statistics (the exact mixing that caused the earlier corruption). When
  `uses_kd=False` there is only one group regardless of `mode`, so this is
  provably identical to the old global behavior for `rank_extension`'s two
  non-KD variants (previously confirmed safe). Also now calls
  `log_classifier_row_norm_diagnostics()` before and after rescaling when
  `method_name` is given.
- **Call sites**:
  - `run_simple_avg_variant()`, `~line 3528-3541`: passes
    `mode=method_cfg.get("calibration_mode", "global")`,
    `uses_kd=bool(method_cfg["uses_kd"])`, `method_name=method_name` when
    `apply_calibration` is True (unchanged net behavior for `simple_avg` —
    `calibration_mode` is always `"global"` for this family); added an
    `else` branch that calls `log_classifier_row_norm_diagnostics(...,
    phase="pre_calibration")` directly so non-calibrated methods still get
    logged (full 8-method coverage in the new diagnostic table).
  - `run_rank_extension_variant()`, `~line 5027-5040`: same pattern, this is
    the site where the fix actually changes behavior for `rank_extension`
    (previously always skipped calibration entirely; now calls
    `calibrate_classifier_row_norms(mode="regime_grouped", uses_kd=...)`).
- **`run_config.json`, `~line 6137`**: added
  `"classifier_calibration_mode_by_family": dict(CALIBRATION_MODE_BY_FAMILY)`
  and `"rankext_family_aware_calibration_enabled":
  bool(RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED)` to the dumped config dict.
- **Startup print block, `~line 1317` and `~line 1588-1590`**: added the new
  flags to the existing `print("\nRank extension:")` dict and the existing
  `print("Classifier calibration by family...")` lines, matching how every
  other flag in this file is surfaced at run start.

**Risk / protocol note:** this changes `rank_extension`'s `apply_calibration`
from `False` to `True` for the first time — its "no calibration" status is
intentionally ending, per the task's explicit request to fix classifier
recency bias. It is still a pure post-hoc, eval-time reweighting of the
already-merged classifier (no retraining, no new parameters, no rehearsal
data) — the same category of change as the calibration already shipped for
`simple_avg`, not a new mechanism (see `report.txt` B.2 for why the
alternative task-ID-probe fix WAS ruled out as a new mechanism).

## 2 — Rank-schedule revert (Task C)

**Why:** `report.txt` Task A.3 — WIDERANK ([32,64,96,128,160]) vs STRICT
([16,32,48,64,80]) is a clean single-lever comparison (both runs share
`rankext_orth_lambda_warmup_enabled=true`). 3 of 4 `rank_extension` variants
got WORSE with wide capacity (net -0.66pp average); doubling trainable rank
did not help. Reverting isolates FIX 1 (item 1 above) as the sole lever vs.
the STRICT baseline in the next run.

- **`USE_RANKEXT_RANK_SCHEDULE_WIDE`, `~line 712`**: `True` → `False`.
  `RANKEXT_RANK_SCHEDULE_WIDE` itself (`[32, 64, 96, 128, 160]`) is left
  defined but unused, matching this file's existing convention for flags with
  a documented but currently-off alternative value. Everything else adopted
  from the WIDERANK run (`RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED=True`, combined
  loss-scale settings, target modules, head LR multipliers) is left
  untouched.

## Verification

- `python -m py_compile vit_lora_cifar100_full5step_n5.py` → clean (no
  output, exit 0).
- `python -m pyflakes vit_lora_cifar100_full5step_n5.py` → clean (zero
  warnings).
- `final_per_method_config_table.py` / `.csv` (this directory): a standalone
  reproduction of `add_method()`'s resolution logic (no torch/dataset
  dependency, since the full script can't be run without a GPU/dataset this
  session) — confirms all 4 `rank_extension` methods now resolve to
  `rank_schedule="16->32->48->64->80"`, `apply_calibration=True`,
  `calibration_mode="regime_grouped"`; all 4 `simple_avg` methods are
  unchanged (`apply_calibration=True`, `calibration_mode="global"`,
  `rank_schedule="fixed:80"`).
- `projected_improvement.csv` (this directory): dry-run projection of FIX 1
  against R5's saved per-step open/restricted accuracy numbers — see
  `report.txt` Task B.5 for methodology and caveats (restricted accuracy is
  a valid upper bound, not a guaranteed outcome, since R5 has no saved
  logits/weights to literally replay the fix against).
