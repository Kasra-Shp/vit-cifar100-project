# Supervisor Experiment 2 -- Preparation Audit

**Date:** 2026-09-15
**Status:** PREPARED, NOT LAUNCHED (training not started, per instruction)

## Summary

This is **Supervisor Experiment 2**: CIFAR-100 5x20, comparing SimpleAvg at
rank 32 against RankExt with cumulative rank schedule `[32,64,96,128,160]`
(+32 rank appended per step). It is a **direct derivative of the already-
COMPLETED Supervisor Experiment 1**
(`experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`,
untouched by this work).

Only capacity/rank-configuration changes are intended relative to Experiment
1. **No R8 regularizer-repair changes are included** -- this file does not
import, port, or reimplement anything from
`experiments_prepared/supervisor_regularizer_repair_both_families_r8.py`
("R8"): no R8-corrected (old-seen-only) KD, no R8 DenseOrth formulation.

## Source of truth

| | Path |
|---|---|
| Parent (Experiment 1, completed, untouched) | `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py` |
| New (Experiment 2, prepared, not launched) | `experiments_prepared/supervisor_exp2_cifar100_5x20_rank32_vs_rankext160.py` |

Experiment 2 was created as a full copy of Experiment 1 (same "dedicated
copy, not a thin import wrapper" convention Experiment 1 itself uses relative
to the canonical `n5.py` -- see that file's own "WHY A FULL COPY" comment),
then surgically edited at exactly the points needed for the three intentional
changes below plus their necessary safety-check/reporting-text follow-through.
A byte-level `diff` between the two files confirms every non-overridden line
is unchanged; the only touched regions are: the top-of-file header, the
`RUN_NAME_BASE` line, `LORA_R`, `RANKEXT_RANK_SCHEDULE`, the dead-code
`RANKEXT_RANK_SCHEDULE_WIDE` companion list, and the Experiment-1-specific
hard-assertion + startup-diagnostics block (updated to Experiment-2 values,
plus new sentinel checks -- see "Safety checks added" below).

## Note on a pre-existing untracked file

`experiments_prepared/vit_lora_cifar100_5x20_rank32_capacity_seed42_EXP2.py`
already existed in the working tree (untracked, no git history) before this
task started. It was **not used** as the basis for this work: it targets all
four attention projections (`q_proj/k_proj/v_proj/out_proj`) rather than the
`q_proj/v_proj` the supervisor specified, and its provenance/derivation from
Experiment 1 could not be confirmed. Per the explicit instruction to use
`supervisor_exp1_cifar100_5x20_fixed_rankext.py` as the direct scientific
parent, this pre-existing file was left untouched and ignored. Flagging this
for the user's awareness, not modifying or deleting it.

## Intended configuration

- Dataset: CIFAR-100 (100 classes)
- Protocol: 5 steps x 20 classes/step
- Seed: 42
- Epochs: 9 (same as Experiment 1, for both LoRA/SimpleAvg and RankExt paths)
- Class order: unchanged (contiguous native-label chunks, same as Experiment 1)
- Target modules: `q_proj`, `v_proj` (both families -- same as Experiment 1's
  `TARGET_MODULES_BY_FAMILY`, which was already `q_proj`/`v_proj` for both
  `simple_avg` and `rank_extension`)

### SimpleAvg
- `LORA_R = 32`, `LORA_ALPHA = 64` (`= 2 * LORA_R`), scaling = 2.0
- Independent fresh adapter per step (unchanged)
- Final merge: dense-delta arithmetic mean (unchanged Experiment-1 mechanism
  -- `extract_lora_state()` -> dense `B@A` deltas -> elementwise mean ->
  `apply_deltas_to_base()`; A/B factors are never averaged directly)
- Classifier rows/biases stitched per introducing step (unchanged)

### RankExt
- Cumulative schedule: `[32, 64, 96, 128, 160]` (+32 new rank per step, every
  step -- verified programmatically, see Safety checks)
- `RANKEXT_ALPHA_PER_RANK = 2.0` (unchanged) -> scaling = 2.0 at every step,
  for both the narrow (Experiment 1) and this wider schedule, by construction
  (`GrowingRankLoRALinear.scaling = rankext_alpha / total_rank =
  RANKEXT_ALPHA_PER_RANK`, `total_rank` cancels out identically regardless of
  its numeric value -- confirmed by reading `GrowingRankLoRALinear` directly,
  not assumed)
- Previous blocks frozen, only the newest block trains per step, all prior
  blocks active in forward (unchanged)

## Methods (same 8 as Experiment 1, NOT renamed into R8 IDs)

1. `simple_avg`
2. `simple_avg_kd_T2`
3. `simple_avg_factor_orth`
4. `simple_avg_factor_orth_kd_T2`
5. `rank_extension`
6. `rank_extension_kd_only_T2`
7. `rank_extension_orth_factor_lam_50`
8. `rank_extension_orth_factor_lam_50_kd_T2`

`METHODS_TO_RUN` / `ACTIVE_METHOD_MAP` / `ACTIVE_METHOD_NAMES` are unedited
from Experiment 1 -- verified live (not just by reading source) via the dry
run described under "Verification performed" below: `Expanded active methods
for this run` printed exactly this 8-method list.

## KD, FactorOrth, RankExt auxiliaries, calibration, best-epoch selection, metrics

None of `IndependentLoraOrthTrainer` / `DeltaOrthRankExtensionTrainer`
(KD + FactorOrth loss computation), the calibration functions
(`calibrate_classifier_row_norms` and family/regime-grouped logic), the
best-epoch (validation-CE) selection logic, the pretrained-backbone feature
anchor / projected-feature-protection / new-block-warmup RankExt auxiliaries,
or the BWT/forgetting/open/restricted metric functions are touched by this
file. They are reused byte-identically from Experiment 1, which itself reuses
them byte-identically from the canonical `n5.py` (per Experiment 1's own
PROVENANCE NOTE, still accurate here). Confirmed by the `diff` referenced
above: none of these regions appear in the changed-line set.

- KD: current-step batch only, no replay, full 100-way logits, no old-class
  mask, T=2, KL(teacher||student), x T^2, weight=1, active from step 2, no
  KD-specific warmup -- unchanged code path.
- FactorOrth: lambda=50, Exp1's warmup behavior, historical averaged-prior
  (SimpleAvg) / frozen-accumulated-prior (RankExt) reference -- unchanged
  code path.
- Calibration: `confidence_weighted_regime_grouped` for both families,
  unchanged grouping/boosts/method detection.
- `forward_transfer` remains absent (not resurrected).

## Static difference audit

| Component | Exp1 | Exp2 | Verdict |
|---|---|---|---|
| Dataset | CIFAR-100 | CIFAR-100 | MATCH |
| Protocol | 5x20 | 5x20 | MATCH |
| Seed | 42 | 42 | MATCH |
| Epochs (LoRA/RankExt) | 9 / 9 | 9 / 9 | MATCH |
| Class order | contiguous native-label chunks | same | MATCH |
| Train/val split | `VALIDATION_PER_CLASS`, same semantics | same (unedited) | MATCH |
| Batch size | `BATCH_LORA=16`, `ACCUM_LORA=1` | same (unedited) | MATCH |
| Optimizer | AdamW | AdamW (unedited) | MATCH |
| Learning rate | `LR_LORA=5e-5`, `LR_RANKEXT=1e-4` | same (unedited) | MATCH |
| Weight decay | 0.05 | 0.05 (unedited) | MATCH |
| Scheduler | cosine | cosine (unedited) | MATCH |
| Target modules | q_proj/v_proj (both families) | same (unedited) | MATCH |
| SimpleAvg rank | 80 | 32 | **INTENTIONAL CHANGE** |
| SimpleAvg alpha | 160 | 64 | **INTENTIONAL CHANGE** |
| SimpleAvg scaling | 2.0 | 2.0 | MATCH |
| SimpleAvg merge | dense-delta arithmetic mean, classifier stitched | same (unedited) | MATCH |
| RankExt schedule | [16,32,48,64,80] (+16/step) | [32,64,96,128,160] (+32/step) | **INTENTIONAL CHANGE** |
| RankExt scaling | 2.0 (`RANKEXT_ALPHA_PER_RANK`) | 2.0 (unedited) | MATCH |
| KD | T=2, weight=1, current-step-only, no mask | same (unedited) | MATCH |
| FactorOrth | lambda=50, Exp1 warmup | same (unedited) | MATCH |
| Auxiliary RankExt mechanisms | anchor/projected-protect/new-block warmup | same (unedited) | MATCH |
| Best-epoch selection | validation CE | same (unedited) | MATCH |
| Classifier handling | per-step stitched (SimpleAvg), masked/protected rows (RankExt) | same (unedited) | MATCH |
| Calibration | confidence_weighted_regime_grouped | same (unedited) | MATCH |
| Open evaluation | 100-way argmax | same (unedited) | MATCH |
| Restricted evaluation | step-local 20-way | same (unedited) | MATCH |
| BWT | family-specific convention | same (unedited) | MATCH |
| Forgetting | RankExt avg-forgetting / SimpleAvg avg_forgetting conventions | same (unedited) | MATCH |
| "Final-rank-matched" invariant assert (`LORA_R == RANKEXT_RANK_SCHEDULE[-1]`) | held (80==80) | **removed** (32 != 160 by design) | **NECESSARY CONSEQUENCE of the two changes above, not a 4th independent lever** -- see next section |

Only the three rows marked **INTENTIONAL CHANGE**/**NECESSARY CONSEQUENCE**
differ. No unintended differences were found.

### Why the final-rank-matched invariant had to be removed

Experiment 1 carried two module-level asserts encoding "SimpleAvg's rank
equals RankExt's final cumulative rank" (`assert LORA_R ==
RANKEXT_RANK_SCHEDULE[-1]` and the matching alpha-equality check) -- true by
construction under Experiment 1's numbers (80 == 80). The supervisor's
Experiment 2 request is explicitly a capacity-**mismatched** comparison
(SimpleAvg final rank 32 vs. RankExt final cumulative rank 160), so this
invariant cannot hold and the two asserts would raise `AssertionError` at
import time if left unchanged. They are removed in Experiment 2 and replaced
with an explicit assert of the opposite fact (`LORA_R !=
RANKEXT_RANK_SCHEDULE[-1]`, with a message explaining the intentional
decoupling) so that a future accidental re-introduction of that invariant
(e.g. a careless merge from Experiment 1) fails loudly instead of silently
passing. This is flagged here exactly per the "STOP and report it instead of
silently modifying" instruction -- it is a structural consequence of changes
#1 and #2, not an independent 4th scientific lever, and no other assert,
training-loop, merge, or metric code was touched to accommodate it (verified:
`GrowingRankLoRALinear`, `extract_lora_state`, `simple_average_deltas`,
`apply_deltas_to_base`, and every KD/FactorOrth/calibration function are all
rank-value-agnostic by construction -- confirmed by reading their source, not
assumed).

## Safety checks added (Experiment-2-specific hard assertions)

All fire at import time, before any dataset load or training:

- CIFAR-100 active (`NUM_CLASSES == 100`), no ImageNet leftover names, no
  "imagenet" in run name
- 5 steps, 20 classes/step, seed 42
- All 8 methods present, exact-set match (not just family-flag match)
- SimpleAvg rank == 32, alpha == 64, scaling == 2.0
- RankExt cumulative ranks == `[32,64,96,128,160]`
- RankExt increment == 32 at **every** step (computed programmatically from
  the active schedule's own diffs, not hardcoded per-step)
- RankExt scaling == 2.0 (`RANKEXT_ALPHA_PER_RANK`)
- q_proj/v_proj targets only, both families
- FactorOrth lambda == 50
- KD T == 2, KD weight == 1
- Calibration == `confidence_weighted_regime_grouped`, both families
- Wide RankExt schedule flag OFF (`USE_RANKEXT_RANK_SCHEDULE_WIDE is False`)
- **New:** no R8 corrected-KD / R8 DenseOrth contamination -- sentinel check
  that R8-unique global names (`uses_dense_orth`, `DenseOrth`,
  `KD_WARMUP_ENABLED`, `KD_BASE_WEIGHT`, `ORTH_WARMUP_ENABLED`,
  `old_seen_class_ids`, `LORA_SCALING`, `RANKEXT_SCALING`, etc., taken from
  reading `supervisor_regularizer_repair_both_families_r8.py` directly) are
  absent from `globals()`, and that the run name contains no
  R8/DenseOrth/old-seen wording
- Run name distinct from Experiment 1 / R7 / R8 / canonical results (asserted
  substring checks plus a structurally different `RUN_NAME_BASE`)

## Verification performed

1. **`python -m py_compile`** on the new script: **PASS**.
2. **Live dry-run of the entire config/assertion pipeline**: the script's
   first ~2590 lines (everything through the hard-assertion block, the
   `METHODS_TO_RUN`/`ACTIVE_METHOD_MAP` construction, and the startup
   diagnostics print) were extracted and executed standalone, stopping
   immediately before `dataset = load_dataset("cifar100")` -- i.e. **zero**
   dataset download, **zero** model instantiation, **zero** training. This is
   a genuine self-test of every constant and every hard assertion (Experiment
   1 itself has no lighter-weight self-test/dry-run mode to reuse -- it was
   deliberately removed from the canonical `n5.py` lineage as ImageNet-only
   scaffolding, per Experiment 1's own "WHY A FULL COPY" comment; no such
   mechanism was reintroduced here for the same reason). Result: **all
   assertions passed**, printed diagnostics confirmed
   `simple_avg_rank=32`, `simple_avg_alpha=64`, `simple_avg_scaling=2.0`,
   `rankext_schedule=[32, 64, 96, 128, 160]`, `rankext_scaling=2.0`,
   `run_name_base='clip_vit_lora_cifar100_5x20_supervisor_exp2_sa32_re160_seed42'`,
   and the expanded active-methods list matched the 8 canonical names
   exactly.
3. **Byte-level diff** against Experiment 1 confirmed no unintended edits
   outside the documented change regions.

No training was launched.

## Output

New, distinct run name: `clip_vit_lora_cifar100_5x20_supervisor_exp2_sa32_re160_seed42`
Writes to a new `results/<run_name>_<timestamp>/` directory (same
`ROOT_RESULTS_DIR`/`BASE_OUTPUT_DIR` mechanism as Experiment 1, unedited) --
cannot collide with Experiment 1, R7, R8, or any canonical result directory.
