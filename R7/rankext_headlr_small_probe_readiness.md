# RankExt Head-LR Small Probe — Readiness Audit

**PREPARE-ONLY task.** No training was run. The probe was NOT submitted to
any cluster (this environment has no cluster/SLURM access — see Section 12).
No canonical file was modified.

## 1. Source files created

- `experiments_prepared/rankext_headlr_small_probe.py` — copy of
  `experiments_prepared/final_9method_5x20_performance_recovery.py` (the
  script that actually produced job4971615's results — confirmed in the
  prior forensic review: its RankExt `restricted_mean` values match
  job4971615's forensic summary exactly), with the modifications in
  Section 6 below.
- `experiments_prepared/slurm/rankext_headlr_small_probe.sbatch` — new
  launcher, distinct job-name/output/error, loops over the LR candidate set
  in-job (bash `for` loop invoking the probe script once per LR value).

## 2. Canonical files unchanged — MD5-verified

| File | MD5 before | MD5 after |
|---|---|---|
| `experiments_prepared/final_9method_5x20_performance_recovery.py` | `048331dac367c5b940b0680f04630eed` | `048331dac367c5b940b0680f04630eed` (unchanged) |
| `experiments_prepared/final_8method_5x20_refined.py` | `4673af2628f3d47473abf0ab26bec657` | `4673af2628f3d47473abf0ab26bec657` (unchanged) |

**PASS.** Neither canonical file was opened for writing at any point; the
probe script was created via `cp` and edited only at its own path.

## 3. New task-scale calibration is inactive

The canonical source (`final_9method_5x20_performance_recovery.py`)
contains **zero** occurrences of `PerTaskScaleCalibration`,
`RankExtCalibratedModel`, `apply_rankext_task_scale_calibration`,
`fit_rankext_task_scale_calibration`, or `collect_rankext_logits_labels` —
confirmed by grep against the canonical file before copying. That
calibration stage was added only in `experiments_prepared/
final_8method_5x20_refined.py`, a completely separate file this probe never
touches. So the probe is safe-by-construction here, not merely by an
assertion.

A belt-and-suspenders runtime guard was still added immediately after the
seed block (verified to execute and pass during the safe-namespace check in
Section 8):

```python
RANKEXT_TASK_SCALE_CALIBRATION_ENABLED = False
for _forbidden_calib_name in (
    "PerTaskScaleCalibration", "RankExtCalibratedModel",
    "apply_rankext_task_scale_calibration", "fit_rankext_task_scale_calibration",
    "collect_rankext_logits_labels",
):
    assert _forbidden_calib_name not in dir(), ...
assert RANKEXT_TASK_SCALE_CALIBRATION_ENABLED is False, ...
```

**PASS** — grep-verified absent from source, and the guard itself was
observed to execute and pass in the live safe-namespace check.

## 4. Old row-norm calibration is active and unchanged

`calibrate_classifier_row_norms_confidence_weighted()` /
`calibrate_classifier_row_norms()`, `CALIBRATION_MODE_BY_FAMILY`,
`CALIBRATION_ENABLED_FAMILIES`, `family_applies_calibration()`,
`family_calibration_mode()` are **not touched anywhere in the diff** (see
Section 7's full diff — none of the 20 hunks fall in this code). The
existing hard assertion `assert CALIBRATION_ENABLED_FAMILIES["simple_avg"]
is True and CALIBRATION_ENABLED_FAMILIES["rank_extension"] is True` is
untouched and was confirmed to still pass live (Section 8).

**PASS.**

## 5. Head-LR values — candidate set and rationale

Historical RankExt baseline (job4971615): `HEAD_LR_MULTIPLIER_BY_FAMILY["rank_extension"] = 1.0`
(confirmed by direct read of the canonical script, lines 996-999 —
`"rank_extension": 1.0`, with a code comment explicitly attributing this to
a documented instability finding: "rank_extension's own head-LR multiplier
(x1.0, its own proven-safe BASELINE ... x10 is unsafe specifically for
rank_extension+factor-orth)").

Per the investigation's own instruction to audit before choosing values
(not sweep blindly), that existing comment is exactly the historical
evidence being referenced: ×10 was previously flagged, in this same
codebase, as a plausible transient-instability amplifier specifically for
RankExt+FactorOrth. This directly motivates staying conservative and close
to 1.0.

**Chosen candidate set: {0.5, 1.0, 2.0}.** 1.0 is the historical baseline
(included as a control); 0.5 and 2.0 are the smallest interpretable
half/double steps on either side. 5.0 was considered but not added to the
default sweep — the combined arm (FactorOrth+KD+protect30) is the most
expensive of the two probe methods, and adding a 4th condition would
meaningfully lengthen an already-multi-hour smoke job for a value whose
main purpose (an explicitly-labeled stress reference) is better served by
10.0 if a supervisor specifically wants one; both are left as optional,
manually-added rows in `LR_MULTIPLIERS` inside the sbatch launcher, not run
by default. 10.0 is deliberately never in the default set, matching the
instruction that it is a stress/reference condition, not a preferred
candidate.

## 6. Full list of changes vs. the canonical script

Exactly 3 categories of change, all documented inline with `# PROBE:`
comments (20 diff hunks total, see the full diff kept alongside this
report's preparation — every hunk falls into one of these 3 categories,
verified by direct review, no unaccounted-for hunk):

1. **Method selection** (`METHODS_TO_RUN`, `EXPECTED_ENABLED_METHOD_FAMILIES`,
   `EXPECTED_METHODS`, and the assertions that check them): SimpleAvg
   entirely OFF; RankExt reduced to exactly
   `rank_extension_factor_orth_lam50_fullkd_T2_protect30` (primary) and
   `rank_extension_factor_orth_lam50` (secondary). `rank_extension` (plain)
   and `rank_extension_fullkd_T2_protect30` (KD without FactorOrth) are OFF
   — out of scope per the investigation's spec. Every assertion that
   referenced a now-inactive method (`ACTIVE_METHOD_MAP["simple_avg_kd_..."]`
   etc.) was replaced with an explicit `assert "..." not in ACTIVE_METHOD_MAP`
   (mirroring the same pattern already used for the T4-arm removal in
   `final_8method_5x20_refined.py`), rather than deleted silently — so any
   future accidental re-enable is still caught, not just skipped.
2. **Scale reduction**: `NUM_STEPS` 5→3 (`PROBE_NUM_STEPS`), `RANKEXT_EPOCHS`
   9→2 (`PROBE_RANKEXT_EPOCHS`). `CLASSES_PER_STEP` stays 20 and
   `class_splits` for the 3 steps that do run are byte-identical to the
   first 3 groups of the canonical 5-step run (`[0..19]`, `[20..39]`,
   `[40..59]`, same order — verified live, not just asserted, in Section 8).
   The RankExt rank schedule (`[16, 32, 48, 64, 80]`) is untouched — the
   probe simply stops indexing it after position 2 (rank grows 16→32→48
   across the 3 probed steps, never reaching 64/80, which is an expected,
   disclosed consequence of testing only 3 of 5 steps, not an algorithm
   change). Every downstream assertion that hardcoded `5`/`9`/`100`/`==9`
   for this protocol was updated to reference `PROBE_NUM_STEPS`/
   `PROBE_RANKEXT_EPOCHS` explicitly (never silently loosened to an
   unconstrained check).
3. **Head-LR sweep wiring**: `RANKEXT_HEADLR_PROBE_MULTIPLIER` env var
   (mirrors the existing `REPLICATION_SEED` env-var convention already in
   this exact file), defaults to `"1.0"` (historical baseline) when unset,
   overrides `HEAD_LR_MULTIPLIER_BY_FAMILY["rank_extension"]` only —
   `["simple_avg"]` is untouched (moot, since SimpleAvg never runs).
   `RUN_NAME_BASE` was changed to embed the probe identity, scale, and swept
   LR value (`rankext_headlr_probe_3x20_2ep_lr{X}_seed42`) so that the
   sbatch launcher's per-LR loop (Section 9) writes each condition to its
   own isolated results/checkpoints directory — this is a real correctness
   fix, not cosmetic: the canonical script's original `RUN_NAME_BASE` did
   not depend on any LR value, so running the same script 3× unmodified
   would have had the 2nd and 3rd LR conditions silently overwrite the
   1st's output directory.

**Everything else — rank schedule, LoRA q/v targets, KD temperature/weight/
scope/warmup for the combined arm, `protect_weight=30.0`,
`RANKEXT_PROJECTED_PROTECT_METHODS`/`RANKEXT_NEW_BLOCK_WARMUP_DISABLED_
METHODS` membership, FactorOrth `lambda_orth=50.0`, classifier restoration,
optimizer type/hyperparameters, scheduler, base LR, row-norm calibration —
is byte-identical to the canonical script.** No line touching any of these
mechanisms appears anywhere in the diff.

## 7. Full diff

20 hunks total (`diff -u final_9method_5x20_performance_recovery.py
rankext_headlr_small_probe.py`), each reviewed and falling cleanly into one
of the 3 categories in Section 6 above — no unexplained/stray hunk. Full
diff available on request; summarized inline above.

## 8. Static verification — actually executed, not just eyeballed

Reused the exact AST-based safe-namespace-extraction technique already
established in `R7/posthoc_calibration/evaluate_posthoc_task_calibration.py`
(excludes the two top-level training-driver `for` loops —
`for method_name in simple_avg_execution_order` /
`for method_name in rank_extension_execution_order` — and everything after
the second one; execs the rest, including real CIFAR-100 dataset loading and
train/val split construction, in a scratch directory). This actually ran
locally (CPU) and **found and required fixing 4 real bugs** that a
read-only line audit alone would have missed, because 4 assertions
elsewhere in the file also hardcoded protocol-scale literals not caught by
the initial edit pass:

1. `assert len(RANKEXT_RANK_SCHEDULE_WIDE) == NUM_STEPS` — checked the
   *unused* WIDE schedule's length against the new reduced `NUM_STEPS`;
   fixed to check against its own definitional length (5), since WIDE stays
   `False` regardless and the schedule actually consumed by training
   (`RANKEXT_RANK_SCHEDULE`) is untouched and long enough either way.
2. `assert NUM_STEPS == 5, "This script is 5x20 ONLY"` (a second,
   independently-written protocol lock elsewhere in the file, separate from
   the one first fixed) — updated to check `PROBE_NUM_STEPS`.
3. `assert class_splits == [<5 groups>]` — updated to check only the first
   `PROBE_NUM_STEPS` groups, confirmed live to still be byte-identical to
   the canonical run's own first-3-groups class IDs/order.
4. `RUN_NAME_BASE` isolation (Section 6, item 3) — the first verification
   run (before this fix) was observed to write its scratch output under the
   canonical script's *own* `RUN_NAME_BASE`-derived path
   (`cifar100_5x20_final_9method_performance_recovery_seed42_...`),
   confirming the collision risk was real, not hypothetical. Re-verified
   after the fix.

**Final verification run output (after all fixes):**
```
[verify] Extracted 569 top-level statements (excluded 2 training for-loops + everything after line 9322).
[verify] Safe namespace built successfully in ~300s -- ALL assertions in the safe subset passed.
[verify] ACTIVE_METHOD_NAMES: ['rank_extension_factor_orth_lam50', 'rank_extension_factor_orth_lam50_fullkd_T2_protect30']
[verify] NUM_STEPS: 3 CLASSES_PER_STEP: 20
[verify] RANKEXT_EPOCHS: 2
[verify] HEAD_LR_MULTIPLIER_BY_FAMILY: {'simple_avg': 10.0, 'rank_extension': 1.0}
[verify] RANKEXT_TASK_SCALE_CALIBRATION_ENABLED: False
[verify] RESULT: PASS
```
(`rank_extension: 1.0` here reflects the default/no-env-var-set behavior —
the historical baseline — confirming the probe is a behavioral no-op vs.
job4971615's own head-LR setting when run without
`RANKEXT_HEADLR_PROBE_MULTIPLIER` set.)

This is genuine empirical confirmation of every assertion reachable before
the training driver: method count/set, KD/protect/FactorOrth mechanics for
the 2 active arms, protocol-scale assertions, calibration-absence guard, and
head-LR wiring. It does **not** exercise the training loop itself (no
gradient step, no checkpoint write, no accuracy computation) — that can only
happen on the cluster.

## 9. Python compiles / launcher syntax-checks

- `python -m py_compile experiments_prepared/rankext_headlr_small_probe.py`
  → **PASS**.
- `bash -n experiments_prepared/slurm/rankext_headlr_small_probe.sbatch` →
  **PASS**.

## 10. Instrumentation decision (spec Section 7-8)

**Deliberately did NOT add new bespoke instrumentation** (raw gradient
norms, classifier-update-magnitude deltas) beyond what the canonical
script's existing, already-proven logging already captures:
`final9_batch_log_df`-equivalent per-batch CSV already records
`method, step, epoch, ce_loss`, plus every orth/factor-orth-related loss
term, already sufficient to read step-boundary CE spikes (group by `step`,
look at the first few epochs of each new step) and per-arm loss trajectories
without any new code. The existing classifier-restoration-diagnostics table
and per-step open/restricted accuracy tables are also untouched and will
populate normally for the 2 active arms.

Reasoning for not adding more: this environment cannot execute real
training (Section 12), so any new instrumentation code could not be
verified end-to-end before being handed to the user for a real cluster run
— and the spec explicitly warns "Do not overcomplicate instrumentation if
it changes execution behavior." Reusing already-load-bearing, already-
tested logging is the conservative choice; a supervisor wanting raw
gradient-norm/update-magnitude numbers specifically would need one small,
separately-reviewed addition before the next cluster submission.

## 11. Deliverables NOT created (require actual training)

`R7/rankext_headlr_small_probe_report.md`,
`R7/rankext_headlr_small_probe_summary.csv`, and
`R7/rankext_headlr_small_probe.png` were **not created** — they require real
training output (per-LR accuracy/loss numbers), which this environment
cannot produce (Section 12).

## 12. Why this environment cannot run the probe

Verified directly (not assumed):
- `sbatch`/`srun`: **not found** on this machine (only the `ssh` binary is
  present, no cluster host config anywhere in the repo).
- Local GPU: NVIDIA GeForce GTX 1650 Ti, **4096 MiB** VRAM (`nvidia-smi`).
- Local PyTorch: `torch==2.10.0+cpu`, `torch.cuda.is_available() == False`.

Actual training in this project runs on a remote SLURM cluster (job IDs
4971615/4972616 etc., `experiments_prepared/slurm/*.sbatch` launchers all
`cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project`), submitted by the
user separately — this environment prepares and statically verifies code,
it does not submit or run cluster jobs.

## 13. Exact command to actually run the probe

```
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
sbatch experiments_prepared/slurm/rankext_headlr_small_probe.sbatch
```

This single job internally loops over `LR_MULTIPLIERS=(0.5 1.0 2.0)`,
running `rankext_headlr_small_probe.py` once per value (each with its own
isolated `RUN_NAME_BASE`), training only the 2 RankExt arms under test over
the reduced 3×20/2-epoch smoke protocol. **Not run in this task.**

---

# FINAL TERMINAL SUMMARY

**RANKEXT HEAD-LR SMALL PROBE**

**HISTORICAL PIPELINE RESTORE**
Reference job: job4971615 (via canonical source
`experiments_prepared/final_9method_5x20_performance_recovery.py`)
New task-scale calibration active: **NO** (absent from source by
construction; runtime guard added and confirmed to pass)
Old row-norm calibration active: **YES** (untouched)
KD unchanged: **YES** (temperature/weight/scope/warmup for the combined arm
untouched)
Protect30 unchanged: **YES** (`RANKEXT_PROJECTED_PROTECT_METHODS`
membership/mechanism untouched)
FactorOrth50 unchanged: **YES** (`lambda_orth=50.0` mechanism untouched)
Classifier restoration unchanged: **YES** (code untouched)

**PROBE**
Methods tested: `rank_extension_factor_orth_lam50_fullkd_T2_protect30`
(primary), `rank_extension_factor_orth_lam50` (secondary)
LR multipliers: 0.5, 1.0 (historical baseline), 2.0 (default set; 5.0/10.0
available as optional manual additions, 10.0 explicitly labeled
stress-only, neither run by default)
Tasks: 3 of 5 (Task1, Task1→2 transition, Task2→3 transition), same seed/
class order as the full run
Epochs: 2 per step (vs. historical 9)
Data reduction: none beyond step/epoch count (no per-class subsampling
added)
Full benchmark run: **NO**

**RESULTS**
NOT RUN — awaiting cluster execution (see Section 12/13 for why and the
exact command).

**RANKEXT LR DECISION**
Historical LR: 1.0
Any alternative promising: NOT RUN — no data yet
Recommended next action: submit
`experiments_prepared/slurm/rankext_headlr_small_probe.sbatch` on the
cluster, then re-run this analysis against its real output
Full-scale LR change justified now: **NO (no probe data yet)**

**FILES**
Probe script: `experiments_prepared/rankext_headlr_small_probe.py`
Launcher: `experiments_prepared/slurm/rankext_headlr_small_probe.sbatch`
Readiness report: `R7/rankext_headlr_small_probe_readiness.md` (this file)
Result report: **NOT CREATED — requires actual training run**
Summary CSV: **NOT CREATED — requires actual training run**
Diagnostic PNG: **NOT CREATED — requires actual training run**

**CANONICAL FILES MODIFIED: NO**
