# CHANGES.md — eval-pipeline diagnostic (Task A) + wide rank-schedule capacity test (Task B)

Session date: 2026-07-21. No training run this session. All verification is
static: `py_compile` + `pyflakes` on the edited file, plus standalone
dry-run scripts in this directory that reimplement the relevant formulas
against synthetic tensors (same pattern as `analysis_strict_review/
dryrun_merge_mechanism_logging.py` last session) — the main script is never
imported, because it executes top-to-bottom as a converted notebook and would
otherwise start real training.

All changes are in `vit_lora_cifar100_full5step_n5.py`. Investigation and
root-cause analysis for Task A live in `analysis_pipeline_audit/report.txt`;
this file only documents the code edits and their verification.

---

## TASK A — closed-set per-step accuracy diagnostic + val-CE plot labeling

**Context**: `analysis_pipeline_audit/report.txt` traces why every method's
step_5 per-step accuracy looks anomalously high (e.g. rank_extension:
0.8%/1.85%/2.2%/7.3%/**96.95%**) and why per-epoch val CE in the convergence
plots doesn't contradict this. Short version: (1) `evaluate_per_step_accuracy`
/`evaluate_model`/`evaluate_seen_step_accuracies` all share one evaluation
mechanism — `compute_metrics()`'s **open, unrestricted argmax over all 100
classes** — which is the correct, standard, *intended* class-incremental
evaluation protocol, not a bug; (2) that shared mechanism is only a fair
cross-step comparison if every class's classifier row is on a comparable
scale, which `calibrate_classifier_row_norms()`'s own docstring already
documents is not guaranteed, and calibration is OFF for the entire
rank_extension family (`CALIBRATION_ENABLED_FAMILIES["rank_extension"] =
False`, a **previously deliberate, previously-tested** revert — see that
function's comment block — not an oversight to "fix" here); (3) the per-epoch
val-CE curves in the convergence plots measure something structurally
different (each step's own local validation split, evaluated with the model
*as it stood during that step's own training*) from the retrospective
per-step accuracy tables (the FINAL model, evaluated on the same class group
after every later step has run) — conflating the two reads as a contradiction
that isn't one once this is made explicit.

Given this, "fix the evaluation pipeline" is implemented as an **additive,
non-destructive diagnostic**, not a change to any existing training or
evaluation number:

1. **`restricted_argmax_accuracy(logits, labels, allowed_class_ids)`** (new
   function, next to `evaluate_single_step_accuracy`). Computes the same
   accuracy definition as `compute_metrics()`, except the argmax candidate
   set is masked down to exactly the eval subset's own classes before
   argmax — this removes the cross-class classifier-row-norm/recency
   confound described above, isolating "does the model's representation for
   this step's own classes still work" from "does this step's classes win
   the open competition against every other class ever trained." Pure numpy,
   no model/training dependency.

2. **`evaluate_per_step_accuracy(model, method_name)`** (modified). Was:
   loop over steps, call `trainer.evaluate(eval_dataset=...)`, log only the
   open-set accuracy. Now: loop over steps, call `trainer.predict(...)`
   once per step (same eval_ds, same model, same batch size — one forward
   pass, not two), compute the open-set accuracy **exactly as before**
   (`argmax(logits, axis=1) == labels`, byte-for-byte the same formula
   `compute_metrics()` uses) AND the new restricted accuracy from the *same*
   logits, then log both. **The pre-existing open-set `per_step_map` return
   value and `per_step_accuracy_rows` accumulator are unchanged in content**
   — same values, same shape, same downstream consumers
   (`per_step_accuracy_by_method.csv`, the two heatmaps, backward_transfer).
   Only a second, additive accumulator (`per_step_accuracy_restricted_rows`)
   and its CSV are new.

3. **New output**: `tables/per_step_accuracy_open_vs_restricted_by_method.csv`
   — long format, one row per (method, step_id), columns `accuracy_open`
   (identical to the existing `per_step_accuracy_by_method.csv` "accuracy"
   column), `accuracy_restricted`, and `recency_bias_gap` =
   `accuracy_open - accuracy_restricted`. A large positive gap at step 5 for
   a given method is exactly the smoking-gun signature the report predicts;
   this table makes it directly measurable in the next run instead of
   inferred from code reading. Not added to `output_checklist.txt`'s
   required list (it's a new optional diagnostic, not a pre-existing
   supervisor deliverable).

4. **Plot/label clarifications** (no data changes): `refresh_live_convergence`
   plot titles and the two `lossgrid("val_ce_loss", ...)` figure titles
   (`validation_ce_loss_by_method.png`, `validation_ce_loss_clean.png`) now
   explicitly state that val CE is "this step's OWN local val split, model as
   of THIS step (in-context, not retrospective)" — directly addressing
   observation (c) in the task (RankExt plain's ~0.2 val CE vs. 0.8% step-1
   accuracy: two different measurements of two different model states, not a
   contradiction).

**What this does NOT do, on purpose**: it does not re-enable calibration for
rank_extension (already tried and reverted for a documented, tested reason —
see `CALIBRATION_ENABLED_FAMILIES`'s comment — re-enabling it here would
silently re-introduce a previously-diagnosed regression); it does not change
`evaluate_model()` (the source of `first_step_accuracy`/`later_steps_accuracy`
/`all_seen_accuracy` in every supervisor table), so **no previously-reported
R3 number is altered by this change**, and it does not touch
`evaluate_seen_step_accuracies`/`evaluate_single_step_accuracy` (used for
rank_extension's true forgetting curve and forward-transfer probes) to keep
the diff surgical and scoped to the exact table the investigation was about.

**Cannot be backfilled for the already-completed R3 run**: no model
checkpoints were saved for `results_strict_20260717_light` (`find ... -iname
"*.safetensors" -o -iname "*.bin" -o -iname "*.pt"` returns nothing) — the
new diagnostic needs a live model to call `.predict()` on, so
`per_step_accuracy_open_vs_restricted_by_method.csv` can only be produced by
a re-run of the training script with this code in place, not by
post-processing R3's saved CSVs. See `analysis_pipeline_audit/report.txt`
for the full list of plots this affects and which ones need a re-run.

**Verification**: `py_compile` + `pyflakes` clean (see bottom of this file).
No dry-run script for this part — `restricted_argmax_accuracy` is pure numpy
array masking, exercised directly against a hand-built 3-class toy case in
`analysis_pipeline_audit/report.txt`'s appendix instead of a separate script,
since it needs no torch/model machinery to verify.

---

## TASK B — wide rank schedule + alpha rescaling capacity test

**Goal** (as specified): test whether rank_extension's underperformance vs.
simple_avg is a capacity bottleneck, by giving it a wider per-step rank
schedule `[32, 64, 96, 128, 160]` (was `[16, 32, 48, 64, 80]`) with alpha
rescaled to preserve the 2:1 alpha/rank ratio, leaving simple_avg and every
other family-conditional setting untouched.

### B1. `USE_RANKEXT_RANK_SCHEDULE_WIDE`: `False` → `True`

One-line flag flip (the wide schedule constant, its shape/monotonicity
asserts, and `active_rankext_rank_schedule()` were already implemented and
were already the single source of truth every consumer reads from —
`get_rank_extension_rank_schedule()`, `build_rank_extension_model()`,
`cfg_df()`'s `lora_rank` column, `run_config.json`'s
`rankext_rank_schedule_active` — so this flip alone is sufficient to make
`build_rank_extension_model`/`get_rank_extension_rank_triplet` use ranks
32→64→96→128→160 at steps 1-5). Comment block above the flag rewritten to
document the capacity-test rationale and that this is no longer isolated
from `RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED` (already `True`, adopted from the
prior strict-review run) — the original "one lever per run" plan doesn't
apply cleanly here since the warmup fix is already baked into the baseline
this run starts from; any accuracy change should be attributed to "wide
schedule on top of the already-adopted warmup fix," not to capacity in
isolation.

### B2. Alpha propagation — verified ALREADY correct by construction, no scaling-logic change needed

`GrowingRankLoRALinear.__init__` already computes
`self.rankext_alpha = RANKEXT_ALPHA_PER_RANK * self.total_rank` and
`self.scaling = self.rankext_alpha / self.total_rank` **per instantiation**,
using that step's own `total_rank` from `get_rank_extension_rank_triplet`.
`total_rank` cancels out of `scaling` algebraically
(`scaling = (ALPHA_PER_RANK * total_rank) / total_rank = ALPHA_PER_RANK`), so
`scaling` is pinned at `RANKEXT_ALPHA_PER_RANK = 2.0` at **every** step
regardless of which schedule is active or which specific rank value is in
play. This means the "keep alpha/rank ratio constant at 2:1, per-step, for
the wide schedule" requirement was already satisfied for the actual training
math before this session touched anything — confirmed by dry-run (see
`dryrun_alpha_propagation.py` part (a)/(b): scaling == 2.0 exactly at ranks
32, 64, 96, 128, 160, and separately at 16, 32, 48, 64, 80 for the
unaffected default schedule). **No change was made to `GrowingRankLoRALinear`
or `get_rank_extension_rank_triplet`.**

### B3. `active_rankext_lora_alpha()` (new function) + two reporting-bug fixes

What DID need fixing was **reporting**, not training math. Two call sites
stamped the global `LORA_ALPHA` (160, a `simple_avg`-only constant used in
its real PEFT `LoraConfig`) onto every method's config row, including
`rank_extension`'s:

- `cfg_df()` (feeds `configs/hyperparameters_by_method.json`,
  `configs/run_config.json`'s CFG-derived tables, and
  `tables/hyperparameter_consistency_check.csv`): `c["lora_alpha"]=LORA_ALPHA`
  for every row.
- `method_config_df` (feeds `tables/method_hyperparameter_summary.csv` via
  `summary_table = method_config_df.merge(...)`):
  `method_config_df["lora_alpha"] = float(LORA_ALPHA)` for every row.

This was a **latent bug**, invisible before this session because
`RANKEXT_ALPHA_PER_RANK * RANKEXT_RANK_SCHEDULE[-1]` (2.0 × 80 = 160)
happened to equal `LORA_ALPHA` (160) under the default schedule — the wrong
formula and the right answer coincided by construction. With
`USE_RANKEXT_RANK_SCHEDULE_WIDE=True`, rank_extension's real effective alpha
at its final rank is `2.0 × 160 = 320`, so leaving these two lines unfixed
would have silently reported "160" in every config CSV/JSON while training
actually ran at effective alpha 320 for rank_extension — exactly the kind of
supervisor-facing config/training mismatch the 2026-07-17 strict audit
flagged elsewhere (see `analysis_strict_audit/audit_report.txt` finding C3)
and that this session's `dryrun_alpha_propagation.py` part (d) explicitly
checks does NOT recur here (`rank_extension`+`wide` → 320.0, all four
`(family, schedule)` combinations match hand-computed expected values).

Fix: added `active_rankext_lora_alpha()` (returns
`RANKEXT_ALPHA_PER_RANK * active_rankext_rank_schedule()[-1]`, i.e. the
family/schedule-correct value at every step by the same cancellation
argument as B2), and made both call sites family-conditional —
`np.where(family == "rank_extension", active_rankext_lora_alpha(), LORA_ALPHA)`
— mirroring the exact pattern `cfg_df()` already used for `lora_rank` one
line above it. `simple_avg`'s `lora_alpha` is untouched (still the global
`LORA_ALPHA` constant, 160, unaffected by anything rank_extension does).

### B4. New assert + `run_config.json` fields

Added, immediately after the existing `assert float(RANKEXT_ALPHA_PER_RANK *
RANKEXT_RANK_SCHEDULE[-1]) == float(LORA_ALPHA)` (which only ever checks the
DEFAULT schedule and is unaffected by the wide-schedule flag): a second
assert that `active_rankext_lora_alpha()` equals
`RANKEXT_ALPHA_PER_RANK * active_rankext_rank_schedule()[-1]` for whichever
schedule is actually active — a reporting-consistency guard against a future
edit reintroducing drift between the two.

`run_config.json` gains three new top-level fields so the effective
rank_extension alpha is visible without cross-referencing the per-method
JSON: `rankext_alpha_per_rank` (2.0), `rankext_lora_alpha_active` (320.0 for
this run), `rankext_more_params_than_simple_avg` (see B5 — **kept literally
tied to `USE_RANKEXT_RANK_SCHEDULE_WIDE`'s boolean value**, which is
technically imprecise per B5's finding below; flagged there rather than
silently making this field "correct" by redefining it, since the task named
this exact field/claim explicitly). A `lora_alpha_note` string field points
readers at this whole cluster of fields so the top-level `lora_alpha: 160`
key isn't misread as applying to rank_extension.

### B5. Parameter-count correction — the exact requested schedule gives PARITY, not "more"

**This does not match the task's stated premise and is flagged rather than
silently absorbed.** The task states: "Document explicitly that rank_ext now
has more total parameters than simple_avg." Checked by
`dryrun_alpha_propagation.py`'s parameter-count section (and reproduced in
`print_final_config_table.py`'s output): with CLIP-ViT-B/16's real
dimensions (hidden=768, 12 encoder layers), trainable LoRA parameter count is
`n_target_modules × final_rank × (in_features + out_features) × n_layers`.

- `simple_avg`: 4 modules × rank 80 × 1536 × 12 = **5,898,240**
- `rank_extension`, OLD default schedule: 2 modules × rank 80 × 1536 × 12 =
  2,949,120 (was strictly LESS than simple_avg, matching the task's framing
  of the old baseline)
- `rank_extension`, NEW wide schedule `[32,64,96,128,160]`: 2 modules ×
  rank **160** × 1536 × 12 = **5,898,240**

`2 × 160 == 4 × 80` — both equal 320 "module-rank units." The requested wide
schedule exactly doubles rank_extension's module-rank product from 160 (2×80)
to 320 (2×160), which lands it at **exact parity** with simple_avg's
320 (4×80), not above it. This is implemented exactly as specified
(`[32, 64, 96, 128, 160]`, alpha 320) because those were explicit, unambiguous
numeric instructions — the schedule was not silently changed to "fix" this —
but the "more total parameters" framing in code comments and
`run_config.json` should be read as "parity, tested as an increase from the
prior under-parameterized baseline," not "strictly more than simple_avg."
If strictly-more-than-simple_avg capacity is still wanted for a future run,
the schedule would need a higher final rank than 160 (e.g. final rank 192
would give 2×192=384 > 320) or KD a target-module count above 2 — a decision
left to the supervisor, not made unilaterally here.

### B6. Untouched, verified via `print_final_config_table.py`'s output

- `LAMBDA_ORTH` (50.0) and both rank_extension factor-orth methods'
  effective `lambda_orth` (50.0, `lambda_orth_scale=1.0`) — unchanged, as
  instructed ("Do NOT touch factor-orth lambda").
- `CALIBRATION_ENABLED_FAMILIES`, `HEAD_LR_MULTIPLIER_BY_FAMILY`,
  `TARGET_MODULES_BY_FAMILY` — unchanged (rank_extension: 2 modules,
  head_lr×1, `apply_calibration=False`; simple_avg: 4 modules, calibration
  on, combined-variant `lambda_orth_scale=0.3`/`kd_weight_scale=0.5` for
  `simple_avg_factor_orth_kd_T2` only).
- `simple_avg` family's `final_rank`/`lora_alpha`/`trainable_lora_params` —
  identical to before (80 / 160 / 5,898,240) in every row of the printed
  table.

---

## Files changed

- `vit_lora_cifar100_full5step_n5.py` — all edits described above (Task A:
  `restricted_argmax_accuracy`, `evaluate_per_step_accuracy`, new CSV output,
  plot title strings. Task B: `USE_RANKEXT_RANK_SCHEDULE_WIDE`,
  `active_rankext_lora_alpha()`, `cfg_df()`, `method_config_df`, new assert,
  `run_config.json` fields).

## Files added (this directory, `analysis_rankext_wide_capacity/`)

- `dryrun_alpha_propagation.py` / `dryrun_output.txt` — Task B numeric
  verification (synthetic tensors, no training).
- `print_final_config_table.py` /
  `final_per_method_config_table_wide_schedule.csv` — Task B final
  per-method config table, parsed from the live edited source file.
- `CHANGES.md` — this file.

Task A's investigation/report is `analysis_pipeline_audit/report.txt`
(separate directory, per the task's deliverable naming).

## Verification

```
python -m py_compile vit_lora_cifar100_full5step_n5.py   # clean
python -m pyflakes vit_lora_cifar100_full5step_n5.py     # clean, exit 0
python analysis_rankext_wide_capacity/dryrun_alpha_propagation.py   # ALL SCALING CHECKS: PASS
python analysis_rankext_wide_capacity/print_final_config_table.py   # 8-row table, alpha=320/160 split confirmed
```

No training was run. No dataset or model checkpoint was loaded or downloaded
this session.
