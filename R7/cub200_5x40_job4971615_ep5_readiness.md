# CUB-200-2011 5×40 job4971615-compatible readiness

Status date: 2026-09-22  
Experiment: **CUB-200-2011, full 200 classes, 5 tasks × 40 classes, seed 42, epochs=5**

## Final verdict

**READY for full-run execution after committing the four new artifacts below.** The requested
readiness checks passed without starting an epoch or submitting SLURM. The cluster must have
the configured shared Hugging Face dataset/model cache populated or outbound access to the
Hugging Face Hub/CDN when the launcher is submitted.

## Dataset source and split verification

- Dataset identifier: `Donghyun99/CUB-200-2011` on the Hugging Face Hub.
- Dataset configuration/revision: `default`, revision `main` (the dataset build loaded on
  2026-09-22).
- Dataset page: https://huggingface.co/datasets/Donghyun99/CUB-200-2011
- Official CUB source reference: http://www.vision.caltech.edu/datasets/cub_200_2011/
- Columns verified: `image`, `label`.
- Image loading verified: a live sample loaded as a PIL/JPEG image (`500×336`).
- Original classes verified: exactly 200, with 200 class names cross-checked against the
  persistent artifact.
- Official splits verified: train `5,994`, test `5,794`.

The official test split remains evaluation-only. Validation is derived only from the official
training split with a deterministic seed-42 per-class shuffle and `5` validation images per
class. Every selected class retains the remaining training images. Final sizes are:

| Split | Images | Construction |
|---|---:|---|
| train | 4,994 | official train minus 5/class validation |
| validation | 1,000 | 5/class from official train only |
| test | 5,794 | official CUB test, untouched |

Per-task counts after the fixed label remap:

| Task | Train | Validation | Test | Classes |
|---:|---:|---:|---:|---:|
| 1 | 999 | 200 | 1,152 | 40 |
| 2 | 998 | 200 | 1,176 | 40 |
| 3 | 998 | 200 | 1,144 | 40 |
| 4 | 1,000 | 200 | 1,166 | 40 |
| 5 | 999 | 200 | 1,156 | 40 |
| **Total** | **4,994** | **1,000** | **5,794** | **200** |

The choice of 5 validation images/class is a `DATASET-REQUIRED` split adaptation: CUB has
relatively few training images per class, so the canonical 25/class setting was not copied
blindly. No test image enters training or validation.

## Persistent class order and protocol

Artifact: `experiments_prepared/splits/cub200_class_order_seed42_5x40.json`

The artifact records the dataset identifier and revision/configuration, seed `42`, original
class count `200`, task count `5`, classes/task `40`, all original class names, both mapping
directions, and the five task blocks. The order is generated once by a deterministic
`random.Random(42)` permutation of original IDs `0..199`; the resulting shuffled order is
remapped to benchmark-local IDs `0..199`.

The task definition is exactly:

- Task 1: new IDs `0..39`
- Task 2: new IDs `40..79`
- Task 3: new IDs `80..119`
- Task 4: new IDs `120..159`
- Task 5: new IDs `160..199`

The script validates the artifact rather than regenerating or overwriting it. All eight method
families use the module-level `class_splits` derived from this same artifact. Seen-class
progression is hard-asserted as `[40, 80, 120, 160, 200]`.

## Active methods and preserved hyperparameters

Exactly these eight methods are active:

1. `simple_avg`
2. `simple_avg_kd_oldseen_T2_warmup`
3. `simple_avg_dense_orth_lam20`
4. `simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup`
5. `rank_extension`
6. `rank_extension_fullkd_T2_protect30`
7. `rank_extension_factor_orth_lam50`
8. `rank_extension_factor_orth_lam50_fullkd_T2_protect30`

Preserved configuration:

- Backbone: `openai/clip-vit-base-patch16`.
- LoRA targets: `q_proj`, `v_proj` for both families.
- SimpleAvg rank/alpha/scaling: `80 / 160 / 2.0`.
- SimpleAvg LoRA LR: `5e-5`; head LR multiplier: `10`.
- SimpleAvg KD: old-seen logits only, current-task images, `T=2`, weight `1.0`, `T²`
  correction, frozen/no-grad teacher, historical warmup.
- DenseOrth: Frobenius cosine-squared formulation, lambda `20`, detached historical
  references.
- RankExt head LR multiplier: `1`.
- RankExt full KD: `T=2`, weight `1`.
- RankExt Protect: `30`.
- RankExt FactorOrth lambda: `50`.
- Calibration: historical `confidence_weighted_regime_grouped` only.
- No active replay, task-scale calibration, T4 arm, new loss, or post-hoc learned scale.

All active family and auxiliary training constants resolve to exactly `5`, including
`LORA_EPOCHS`, `RANKEXT_EPOCHS`, `FT_EPOCHS`, `JOINT_EPOCHS`, `ORTH_EPOCHS`,
`FULL_*_EPOCHS`, and `SCRATCH_EPOCHS`.

RankExt is unchanged in depth and capacity: `[16, 32, 48, 64, 80]`, increments
`[16, 16, 16, 16, 16]`, final rank `80`.

## Classifier, KD, calibration, merge, and metrics audit

The classifier is constructed with `num_labels=NUM_CLASSES`, and the smoke test verified
`(1, 200)` logits for both a SimpleAvg model and a RankExt model. Old-seen/current-task/full
logit masks, class stitching, task groups, restricted evaluation, all-seen evaluation, and
checkpoint logic consume `NUM_CLASSES`, `NUM_STEPS`, and `CLASSES_PER_STEP` rather than a
100-class or 20-class protocol literal. The existing BWT, forgetting, per-task accuracy,
open/all-seen accuracy, restricted/task-oracle accuracy, and RankExt `R_ij` reporting paths
remain in the canonical structure; SimpleAvg does not receive a fabricated persistent `R_ij`.

## Stale-hardcoding audit

The active protocol assertions are `NUM_CLASSES=200`, `NUM_STEPS=5`, and
`CLASSES_PER_STEP=40`. Active class construction and loops are generic over those constants.
The audit found:

- No active `range(100)`, `NUM_CLASSES == 100`, or `CLASSES_PER_STEP == 20` protocol logic.
- Historical `100-way`, `5×20`, and `20-row` text remains in copied provenance comments and
  docstrings only; it does not control labels, masks, output dimensions, schedules, or loops.
- `REPLAY_PER_CLASS=20` and the wide `[32,64,96,128,160]` schedule remain as disabled legacy
  configuration values inherited from the canonical file; hard assertions confirm that no
  active method uses replay and that `USE_RANKEXT_RANK_SCHEDULE_WIDE=False`.
- The disabled T4 flag remains explicitly false and is asserted absent from the active map.

These are `IMPLEMENTATION-NECESSARY`/historical compatibility declarations, not dangerous
stale protocol assumptions.

## Difference classification

### DATASET-REQUIRED

- Load CUB-200-2011 rather than CIFAR/ImageNet-100.
- Preserve its official train/test split and `image`/`label` schema.
- Deterministic label remap from CUB original IDs to the persistent seed-42 order.
- Deterministic `5`-per-class validation split from official training data only.
- CLIP-compatible image preprocessing through the processor/transform path.
- Indexed classwise split construction to avoid repeatedly rescanning the 200-class CUB table;
  this preserves the same deterministic classwise split semantics.

### IMPLEMENTATION-NECESSARY

- Generalize classifier dimensions, class masks, task bounds, stitching, calibration groups,
  and metric loops to `200` classes and `40` classes/task.
- Add hard checks for artifact coverage, task coverage, seen-class progression, output shape,
  and exact active method/epoch/schedule configuration.
- Use the requested shared-cache launcher paths.

### REPORTING-ONLY

- New CUB run name, output paths, dataset/protocol labels, and CUB-specific split-count rows.
- Existing five-task and 5×5 reporting structures are retained.

### SCIENTIFIC-DIFFERENCE

Only the requested differences are present:

1. CUB-200-2011 instead of the canonical CIFAR/ImageNet-100 dataset.
2. `200` total classes instead of `100`.
3. `40` classes/task instead of `20`, while retaining five tasks.
4. `5` epochs instead of the canonical source run's `9` epochs.

Preserved are the five-step sequence depth, RankExt schedule/final rank, eight methods, losses,
KD semantics, calibration mode, replay policy, merge semantics, classifier restoration, and
evaluation semantics. No additional scientific difference was found.

## Readiness checks

| Check | Result |
|---|---|
| Python compile | PASS |
| SLURM shell syntax (`bash -n`) | PASS |
| Dataset metadata/source load | PASS |
| Original class count = 200 | PASS |
| Official train/test sizes = 5994/5794 | PASS |
| Persistent artifact and deterministic seed-42 order | PASS |
| Full 200-class coverage/no duplicates | PASS |
| Five tasks × 40 classes | PASS |
| Seen progression `[40,80,120,160,200]` | PASS |
| Classifier output dimension 200 | PASS |
| Train-only validation/no test leakage | PASS |
| CLIP processor and 224×224 sample preprocessing | PASS |
| CLIP ViT-B/16 model load | PASS |
| Minimal SimpleAvg forward/backward | PASS |
| Minimal RankExt forward/backward | PASS |
| Exactly eight active methods | PASS |
| Every active epoch setting = 5 | PASS |
| Active RankExt schedule `[16,32,48,64,80]` | PASS |
| No T4 arm/task-scale calibration | PASS |
| No full epoch/training driver | PASS (not started) |

The model loader reported unused text-side checkpoint weights while loading the vision-only
CLIP model; this is expected for `CLIPVisionModel`, and the vision weights loaded successfully.

## Experimental-design rationale

> CUB-200-2011 is evaluated using all 200 classes in five 40-class incremental tasks.
> Keeping the number of incremental steps fixed at five preserves the continual sequence depth
> and RankExt growth schedule of the CIFAR-100 benchmark, while increasing the number of
> fine-grained classes presented per task.

This is **not** directly matched to CIFAR-100 in total class count; the controlled quantity is
sequence depth (five tasks), not total number of classes.

## Files created

- `experiments_prepared/final_8method_cub200_5x40_job4971615_ep5.py`
- `experiments_prepared/slurm/final_8method_cub200_5x40_job4971615_ep5.sbatch`
- `experiments_prepared/splits/cub200_class_order_seed42_5x40.json`
- `R7/cub200_5x40_job4971615_ep5_readiness.md`

No canonical CIFAR-100 or ImageNet source file was modified. Full training was not started and
SLURM was not submitted.
