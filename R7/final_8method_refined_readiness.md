# Final 8-Method Refined Experiment — Readiness Audit

No training was run for this task beyond safe, no-training static checks (an AST-based safe-namespace
extraction that excludes both top-level training-driver `for` loops — the same trusted technique
already used in `R7/posthoc_calibration/evaluate_posthoc_task_calibration.py`). No canonical file was
modified.

## 1. Source files copied

- `experiments_prepared/final_9method_5x20_performance_recovery.py` → `experiments_prepared/final_8method_5x20_refined.py`
- `experiments_prepared/slurm/final_9method_5x20_performance_recovery.sbatch` → `experiments_prepared/slurm/final_8method_5x20_refined.sbatch`

## 2. Original files unchanged

Verified by MD5 checksum, before and after all edits:

| File | MD5 |
|---|---|
| `final_9method_5x20_performance_recovery.py` | `048331dac367c5b940b0680f04630eed` (unchanged) |
| `slurm/final_9method_5x20_performance_recovery.sbatch` | `a22b7b7b686a9bb98fba60388a741986` (unchanged) |

**PASS.**

## 3. T4 removal

Removed the `simple_avg_kd_oldseen_T4_warmup` arm's `add_method(...)` registration, its
`METHODS_TO_RUN`/`EXPECTED_ENABLED_METHOD_FAMILIES`/`EXPECTED_METHODS`/`_SA_KD_ARMS` entries, its
dedicated temperature assertion, the now-unnecessary `_T4_ABLATION_BASE_METHODS` KD-sweep special
case, its automatic within-run comparison rows, and its cross-run reference-table row. Replaced with
an explicit assertion (`"simple_avg_kd_oldseen_T4_warmup" not in ACTIVE_METHOD_MAP`) so any future
accidental re-introduction fails loudly rather than silently.

**PASS** — verified by the safe-namespace checker: `no_T4_arm` = PASS.

## 4. Final method list

```
simple_avg
simple_avg_kd_oldseen_T2_warmup
simple_avg_dense_orth_lam20
simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup
rank_extension
rank_extension_fullkd_T2_protect30
rank_extension_factor_orth_lam50
rank_extension_factor_orth_lam50_fullkd_T2_protect30
```
8 methods (4 SimpleAvg + 4 RankExt), confirmed live via the safe-namespace checker (`ACTIVE_METHOD_NAMES`,
`EXPECTED_METHODS`, both size 8, both equal).

## 5. Head-LR evidence and final choice

Full audit in `R7/head_lr_final_decision.md`. Summary: RankExt ×10 has documented, dated evidence of
being a plausible instability amplifier for FactorOrth configurations (never claimed sufficient alone);
SimpleAvg's ×10 has no controlled evidence of necessity, only absence-of-observed-harm. **Chosen: a
single shared `HEAD_LR_MULTIPLIER = 1.0` for both families.** This is not a dedicated LR ablation —
limitation disclosed in that report.

## 6. Shared head-LR implementation

`HEAD_LR_MULTIPLIER_BY_FAMILY` dict removed entirely; `family_head_lr_multiplier(family)` now always
returns the single `HEAD_LR_MULTIPLIER` constant regardless of `family`. Verified live:
`family_head_lr_multiplier("simple_avg") == family_head_lr_multiplier("rank_extension") == 1.0`, and
every entry in `ACTIVE_METHOD_CONFIGS` resolves `head_lr_multiplier == 1.0`.

**PASS.**

## 7. RankExt calibration implementation

Added, isolated, in a single new block (not scattered): `PerTaskScaleCalibration` (nn.Module, 4
trainable scalars), `collect_rankext_logits_labels`, `fit_rankext_task_scale_calibration`,
`RankExtCalibratedModel` (inference-only wrapper), `apply_rankext_task_scale_calibration` (orchestrator).
Wired into `finalize_rank_extension_arm()` — the function that actually produces every RankExt arm's
reported final model in this script's active execution path (`run_rank_extension_variant()` is
preserved-but-unused, matching the codebase's existing convention) — immediately after the existing
`calibrate_classifier_row_norms_confidence_weighted()`/`calibrate_classifier_row_norms()` dispatch
(kept fully unchanged) and after the drift diagnostic. Gated by a single flag,
`RANKEXT_TASK_SCALE_CALIBRATION_ENABLED = True`. Fresh scales are fit per-checkpoint every call — no
job-4970580 diagnostic values are hard-coded anywhere (`no_hardcoded_job4970580_scales` = PASS).

## 8. Validation-only fitting

`apply_rankext_task_scale_calibration()` builds its fitting data exclusively from
`make_val_dataset(all_classes)` and passes only `(val_logits, val_labels)` into
`fit_rankext_task_scale_calibration()`. Verified live by source inspection
(`fit_call_uses_val_logits_only`, `val_dataset_uses_make_val_dataset` = PASS).

**PASS.**

## 9. No test tuning

The only place the test set (`eval_all_seen`) is touched by the new stage is *after* the calibration
module's parameters are already fit and frozen — purely to produce the auditability diagnostics
(pre/post open accuracy, restricted-invariance check). No test label or test accuracy ever feeds back
into `a_1..a_4`'s optimization.

**PASS** (by construction/code inspection — see Section 7's call ordering).

## 10. Model weights untouched

`apply_rankext_task_scale_calibration()` explicitly sets `requires_grad=False` on every parameter of
the incoming (already row-norm-calibrated) model before doing anything else. The calibration-fitting
loop (`fit_rankext_task_scale_calibration`) operates purely on detached NumPy arrays converted to
plain tensors — the model is never called with gradients enabled during fitting; only `module.a`
(the calibration's own 4 scalars) ever has `requires_grad=True`, and even that is explicitly frozen
(`requires_grad=False`) again immediately after fitting completes, before the module is ever used for
evaluation. Verified live: `fit_returns_frozen_module` = PASS (0 trainable params in the returned
module), `fit_only_4_params` = PASS.

**PASS.**

## 11. Restricted invariance

`apply_rankext_task_scale_calibration()` computes restricted accuracy per task from the same cached
test logits both before and after calibration and asserts `max |diff| < 1e-6`, printing a loud
warning (not a hard crash, so one arm's anomaly doesn't kill an 18-hour multi-arm run) if violated.
The synthetic smoke test in the safe-namespace checker exercises the identity-init and post-fit
scale-fixing behavior (`calibration_identity_at_init`, `task5_scale_exactly_1_at_init`,
`fit_task5_scale_stays_1` = all PASS); the invariance property itself is a direct mathematical
consequence of per-group positive multiplicative scaling (already empirically confirmed at 0.000e+00
in the separate, already-completed job 4970580 diagnostic run, same equation).

**PASS** (structural/synthetic verification; full empirical re-verification only happens once this
script actually trains a real RankExt arm).

## 12. Output/report schema

No new method name, no `_calibrated` suffix, no 9th column in the main summary/within-run/cross-run
tables — the calibration diagnostics (`rankext_task_scale_calibration_diagnostic_rows`) are a
separate, isolated accumulator, never merged into `training_merge_summary_rows`/`final9_summary_df`
(the script's Python variable names were kept as `final9_*` internally for minimal-diff safety, but
their file output was renamed to `final_8method_*` — see Section 14).

**PASS** (by code inspection — the diagnostics list is appended to exactly once, from inside
`apply_rankext_task_scale_calibration`, and is never referenced anywhere near the main summary-table
construction code).

## 13. SLURM launcher

`experiments_prepared/slurm/final_8method_5x20_refined.sbatch` points `SCRIPT=` at the new
`final_8method_5x20_refined.py`, uses a distinct `--job-name`/`--output`/`--error`, and `bash -n`
syntax-checked clean.

**PASS.**

## 14. Exact files to commit

- `experiments_prepared/final_8method_5x20_refined.py` (new)
- `experiments_prepared/slurm/final_8method_5x20_refined.sbatch` (new)
- `R7/head_lr_final_decision.md` (new)
- `R7/final_8method_refined_readiness.md` (new, this file)

(`experiments_prepared/final_9method_5x20_performance_recovery.py` and its `.sbatch` are unchanged and
need no re-commit.)

## 15. Exact command that would submit the job later

```
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
sbatch experiments_prepared/slurm/final_8method_5x20_refined.sbatch
```

**Not run in this task.**

## Head-LR external literature check

Status: **PASS** (check performed; no reversal required)
Conclusion: A focused literature check (Mittal et al. "Essentials for Class Incremental Learning" CVPR-W
2021; Zhao et al. "Weight Aligning" CVPR 2020; Ahn et al. "SS-IL" ICCV 2021 — full citations and detail
in `R7/head_lr_final_decision.md`'s new "External literature check" section) confirms the literature
strongly supports controlling classifier update magnitude / old-vs-new score-and-norm balance in CIL,
and its one directly-relevant LR-specific finding (LowLR reducing task-recency bias) points the same
direction as our choice (lower, not higher, incremental-step LR reduces bias). No paper establishes a
universal numeric classifier-head-LR multiplier, and none tests a setup resembling ours. **Selected
multiplier unchanged: `HEAD_LR_MULTIPLIER = 1.0` for both families** — the repository's own
project-specific instability evidence remains the primary basis; literature is corroborating, not
overriding.

## Final PNG results pipeline

Renderer implemented: **YES** — `experiments_prepared/render_final_8method_results.py` (new, isolated
module; imports/compiles cleanly; not imported by or modifying any canonical file).
Synthetic render smoke: **PASS** — a clearly-marked mock 8-row dataframe was rendered to a scratch
directory (`R7/_render_smoke_scratch/`, never the real output path) producing a valid, non-empty CSV
(431 bytes) and two valid, non-empty 300 DPI PNGs (table: ~174 KB; bar chart: ~207 KB), both visually
inspected and confirmed clean (light background, no clipping, bold highest all-seen value, correct
em-dash for SimpleAvg's structurally-unavailable Forgetting cells). Three negative-path smoke tests
also confirmed the validator raises (does not silently render) on: a missing arm, an unexpected extra
row (e.g. a stray calibration-method row), and a T4-arm-present summary. All mock outputs and the
scratch directory were deleted immediately after (`R7/_render_smoke_scratch/` no longer exists) — no
mock file can be confused with a real result.
Automatic end-of-run hook: **WIRED, code-verified, not yet end-to-end exercised** — a call to
`render_final_8method_results(final9_summary_df, output_dir="R7")` is inserted into
`final_8method_5x20_refined.py`'s reporting section immediately after the real
`final_8method_summary_table.csv` is written, importing the renderer module via the script's own
directory (robust to how the script is invoked). This call site is syntax-checked and directly
inspected against the renderer's real function signature, but — honestly disclosed — it can only be
exercised end-to-end once the reporting section actually runs after real training, since it depends on
`final9_summary_df`, a training-time-populated global. It deliberately **raises (fails loudly)**,
aborting the reporting section, if the summary does not contain exactly the 8 expected arms.
Requires all 8 methods: **YES** (enforced by `validate_summary_before_render()`, exercised directly in
the synthetic smoke test above).
Real results hard-coded: **NO** (renderer reads only from the passed-in/CSV-loaded dataframe).
Calibration shown as separate method/column: **NO** (an extra row named e.g.
`rank_extension_calibrated` is explicitly rejected by the validator, tested above).
Expected real output: `R7/final_8method_results_table.png` (plus `.csv` and the optional
`R7/final_8method_allseen_accuracy.png`), produced automatically the next time
`final_8method_5x20_refined.py` completes a real run with all 8 arms present.

---

# FINAL TERMINAL SUMMARY

**FINAL 8-METHOD PREPARATION**
T4 variant removed: **YES**
Final method count: **8** (4 SimpleAvg + 4 RankExt)

**HEAD LR**
Old SimpleAvg multiplier: **10.0**
Old RankExt multiplier: **1.0**
Evidence-based common multiplier: **1.0** (both families)
Reason: RankExt ×10 has documented, dated evidence of amplifying a training instability for
FactorOrth configurations; SimpleAvg's ×10 has no controlled evidence of necessity, only
absence-of-observed-harm; ×1.0 is the safer common choice and removes an un-ablated asymmetry.

**RANKEXT CALIBRATION**
Integrated into existing RankExt pipeline: **YES** (inside `finalize_rank_extension_arm`, after the
existing row-norm calibration)
Separate method created: **NO**
Separate main-results column created: **NO**
Validation-only fit: **YES**
Free parameters: **4**
Task-5 reference: **YES, fixed at scale 1.0**
Test tuning: **NO**
Backbone modified: **NO**
LoRA modified: **NO**
Classifier weights modified: **NO**
Restricted invariance: **PASS** (structural/synthetic verification; empirical confirmation pending an
actual training run)

**FILES**
New experiment: `experiments_prepared/final_8method_5x20_refined.py`
New launcher: `experiments_prepared/slurm/final_8method_5x20_refined.sbatch`
Head-LR report: `R7/head_lr_final_decision.md`
Readiness report: `R7/final_8method_refined_readiness.md`
Original canonical files unchanged: **YES** (MD5-verified)

**FULL TRAINING SUBMITTED: NO**
**READY TO SUBMIT: YES**
