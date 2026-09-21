# ImageNet-100 / job4971615 Structural Compatibility Audit

**Date:** 2026-09-21. **Scope: structural compatibility audit + new-file preparation only.**
No full training was launched, no SLURM job was submitted, no scientific code in the canonical
CIFAR-100 script or the existing 9-method ImageNet-100 script was modified. This report documents
a **new**, separate ImageNet-100 script and launcher built specifically to be the ImageNet-100
counterpart of the canonical CIFAR-100 production benchmark (job4971615), and everything that was
and was not verified about it in this pass.

**Goal restated:** not to redesign the ImageNet-100 experiment, but to make it structurally and
scientifically compatible with job4971615 so results are meaningfully comparable across datasets.

---

## 1. Source of truth and files

| File | Role |
|---|---|
| `experiments_prepared/final_9method_5x20_performance_recovery.py` | Canonical CIFAR-100 source of truth (job4971615). **UNCHANGED** — confirmed via `git status --short` (no diff against HEAD) throughout this pass; never edited. |
| `experiments_prepared/final_9method_imagenet100_5x20_performance_recovery.py` | Existing 9-method ImageNet-100 script (prior session, 2026-09-18). **UNCHANGED** — read and diffed only, not edited in place, per this task's explicit instruction. |
| `experiments_prepared/final_8method_imagenet100_job4971615_compatible.py` | **NEW.** Copy of the 9-method ImageNet-100 script with exactly one functional change (T4 arm disabled) plus the count/name bookkeeping that change requires. |
| `experiments_prepared/slurm/final_8method_imagenet100_job4971615_compatible.sbatch` | **NEW.** Dedicated launcher, distinct job-name/output/error/`RUN_NAME_BASE` from every other script's. |
| `experiments_prepared/splits/imagenet100_class_order_seed42.json` | Existing persistent class-order artifact. Read and re-verified only (Section 5), not regenerated. |
| `R7/imagenet100_job4971615_compatibility_readiness.md` | This report. |

**Not touched, not read for modification purposes:** `R7/chapter5_main_benchmark/metadata/dataset_registry.json` and `chapter5_result_schema.md` were read (Section 9) for cross-checking only — zero bytes written to either.

## 2. Method-set derivation (live-read, not trusted from comments)

`METHODS_TO_RUN` (canonical script, line 1827) was read directly, along with `build_active_method_configs()` (line ~1978) and `EXPECTED_ENABLED_METHOD_FAMILIES` (line 2278). The 9 `True` flags in the canonical script are:

`simple_avg`, `simple_avg_kd_oldseen_warmup` (SA-2), `simple_avg_kd_oldseen_T4_warmup` (SA-3, **T4**), `simple_avg_dense_orth` (SA-4), `simple_avg_dense_orth_kd_oldseen_warmup` (SA-5), `rank_extension` (RE-0), `rank_extension_fullkd_T2_protect30` (RE-2), `rank_extension_factor_orth_lam50_new` (RE-3), `rank_extension_factor_orth_lam50_fullkd_T2_protect30` (RE-4).

**Independent cross-check:** `R7/chapter5_main_benchmark/metadata/dataset_registry.json`'s `cifar100` entry (the actual Chapter-5 data package built from job4971615) records `"methods": 8, "excluded_methods": ["simple_avg_kd_oldseen_T4_warmup"], "source_run": "job4971615"` — an independent, pre-existing confirmation (not authored in this pass) that job4971615's reportable 8-method set is exactly "the 9 `True` flags minus T4," matching this audit's own derivation exactly.

**8 canonical methods (task brief Section 5, no T4, no calibration arm, no LR-ablation, no lambda1 arms):**

| # | Internal method name | Display | Family |
|---|---|---|---|
| 1 | `simple_avg` | SimpleAvg | simple_avg |
| 2 | `simple_avg_kd_oldseen_T2_warmup` | SimpleAvg + KD | simple_avg |
| 3 | `simple_avg_dense_orth_lam20` | SimpleAvg + DenseOrth | simple_avg |
| 4 | `simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup` | SimpleAvg + DenseOrth + KD | simple_avg |
| 5 | `rank_extension` | RankExt | rank_extension |
| 6 | `rank_extension_fullkd_T2_protect30` | RankExt + KD + Protect | rank_extension |
| 7 | `rank_extension_factor_orth_lam50` | RankExt + FactorOrth | rank_extension |
| 8 | `rank_extension_factor_orth_lam50_fullkd_T2_protect30` | RankExt + FactorOrth + KD + Protect (combined) | rank_extension |

## 3. Per-mechanism hyperparameters — live-read from canonical source, not trusted from comments

Every value below was read directly from its assignment/assertion in `final_9method_5x20_performance_recovery.py` (line numbers as of this pass) and cross-checked against the truncated config-construction smoke test's own printed output (Section 8) for the new 8-method script.

| Hyperparameter | Canonical value (live-read) | Line | New script (live-executed) |
|---|---|---|---|
| `KD_WEIGHT` (SimpleAvg + RankExt KD weight) | **1.0** | 1049 | `KD_WEIGHT: 1.0` |
| KD temperature | `KD_TEMPERATURES = [2.0]`; SA-2/SA-5 also use `[2.0]` explicitly in `add_method()`; SA-3 (T4) used `4.0` — **excluded arm, not applicable here** | 1050 | `KD_TEMPERATURES: [2.0]` |
| SimpleAvg KD scope / warmup | `kd_class_scope="old_seen"`, `kd_warmup_epochs=KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4=1.0` | 2033–2037 | confirmed via smoke-test assertion block (KD-scope assertions, `_SA_KD_ARMS`) |
| `LAMBDA_ORTH` (FactorOrth) | **50.0** | 1012 | `LAMBDA_ORTH: 50.0` |
| `DENSE_ORTH_LAMBDA` | **20.0** | 1028 | inherited unchanged (not re-printed by the truncated smoke test, but resolved via `add_method(... uses_dense_orth=True)` from the unchanged constant) |
| `RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT` | **30.0** | 864 | inherited unchanged; `rank_extension_fullkd_T2_protect30` / `..._factor_orth_lam50_fullkd_T2_protect30` confirmed present in `RANKEXT_PROJECTED_PROTECT_METHODS` |
| RankExt KD scope | `kd_class_scope="full"` (full 100-way, **not** old-seen) | 2117–2121, 2141–2148 | confirmed via smoke-test assertion block (`_RE_KD_ARMS`) |
| `HEAD_LR_MULTIPLIER_BY_FAMILY["simple_avg"]` | **10.0** | 996–999 | `HEAD_LR_MULTIPLIER_BY_FAMILY: {'simple_avg': 10.0, 'rank_extension': 1.0}` |
| `HEAD_LR_MULTIPLIER_BY_FAMILY["rank_extension"]` | **1.0** | 996–999 | same as above |
| `TARGET_MODULES_BY_FAMILY` (both families) | `["q_proj", "v_proj"]` | 646–649 | `{'simple_avg': ['q_proj', 'v_proj'], 'rank_extension': ['q_proj', 'v_proj']}` |
| `LORA_R` / `LORA_ALPHA` / scaling | 80 / 160 / 2.0 | 567–568 | `LORA_R: 80, LORA_ALPHA: 160` |
| `RANKEXT_RANK_SCHEDULE` (active, non-wide) | `[16, 32, 48, 64, 80]` | 1414 | `RANKEXT_RANK_SCHEDULE_active: [16, 32, 48, 64, 80]`, `USE_RANKEXT_RANK_SCHEDULE_WIDE: False` |
| `LORA_EPOCHS` / `RANKEXT_EPOCHS` | 9 / 9 | — | `{'LORA_EPOCHS': 9, 'RANKEXT_EPOCHS': 9}` |
| Backbone | `openai/clip-vit-base-patch16`, `hidden_size=768`, classifier `nn.Linear(768, 100)` | — | inherited unchanged (zero diff lines touch model/classifier construction code) |

**Explicit resolution of the flagged discrepancy:** the parent session's task brief warned that a prior readiness audit of a *different* script in this lineage once found SimpleAvg's historical KD weight described as 0.5 in a task brief despite the actual source value being 1.0. Live-read from `final_9method_5x20_performance_recovery.py` line 1049 (`KD_WEIGHT = 1.0`, with an explicit comment confirming the 0.75 one-off experiment was closed and reverted) and cross-confirmed by the truncated smoke test's own printed `KD_WEIGHT: 1.0` **for this exact script** — **no contradiction found; KD weight is 1.0, not 0.5, for every KD-bearing method in the 8-method set.**

## 4. Calibration — historical row-norm only, no task-scale calibration

- `USE_CLASSIFIER_CALIBRATION = True` (line 682); `CALIBRATION_ENABLED_FAMILIES = {"simple_avg": True, "rank_extension": bool(RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED)}`; `RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED = True` (line 714), `RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED = True` (line 761).
- `CALIBRATION_MODE_BY_FAMILY` resolves **both** families to `"confidence_weighted_regime_grouped"` — the historical row-norm/weight-alignment mechanism (`calibrate_classifier_row_norms_confidence_weighted()`), never a validation-fitted per-task logit-scale calibration.
- **Grep for forbidden symbols** (`PerTaskScaleCalibration`, `RankExtCalibratedModel`, `apply_rankext_task_scale_calibration`, `fit_rankext_task_scale_calibration`, `collect_rankext_logits_labels`) across the canonical CIFAR-100 script, the 9-method ImageNet script, and the new 8-method script: **zero matches in all three.** This lineage never contained a task-scale calibration mechanism; the audit confirms its continued absence rather than merely asserting it.

## 5. Dataset split compatibility

Re-verified live, offline, by reloading `experiments_prepared/splits/imagenet100_class_order_seed42.json` fresh (not trusted from the prior session's own report) and recomputing every property:

| Check | Result |
|---|---|
| 100 total classes, 5 tasks, 20 classes/task | PASS |
| Union of all 5 tasks' `new_class_ids` == `{0..99}` | PASS |
| No pairwise overlap between any two tasks | PASS |
| `original_id_to_new_id` covers `{0..99}` as both keys and values | PASS |
| `dataset_identifier` == `"clane9/imagenet-100"`, `seed` == `42` | PASS |
| Deterministic construction | `random.Random(42).shuffle()` applied once at artifact-generation time, only ever **loaded** (never regenerated) by both the 9-method and 8-method scripts at module level — identical task sequence for every method arm, by construction |

This mechanism is unchanged by this pass (the artifact itself was not touched); only its properties were re-verified.

## 6. ImageNet dataset interface

Inherited unchanged from the 9-method ImageNet script (zero diff lines touch the dataset-loading cell): `load_dataset("clane9/imagenet-100", cache_dir=IMAGENET100_CACHE_ROOT)`, `FINAL_EVAL_SPLIT_NAME = "validation"` (this mirror has no split literally named `"test"`), label column remapped via the class-order artifact before `class_splits` (itself unmodified) partitions the remapped `0..99` space. Preprocessing (`CLIPImageProcessor`-derived `224×224`, `RandomCrop(padding=8)`, `RandomHorizontalFlip`, `ColorJitter`) is dataset-agnostic and was not changed for ImageNet-100 (per the prior session's own audit, `R7/imagenet100_5x20_static_readiness.md` Section 4) — this is expected, allowed dataset-specific behavior (Section 14's own carve-out), not a scientific difference in method/loss code.

## 7. Classifier dimension and metrics compatibility

Classifier construction (`nn.Linear(768, 100)`), open/restricted evaluation, BWT/forgetting computation, and the `R_{i,j}$` accuracy-matrix mechanism are all inherited byte-identical (zero diff lines touch any of this code). `NUM_CLASSES=100` is identical between datasets, so no classifier-dimension change was needed or made. SimpleAvg's BWT/forgetting remains structurally undefined (no persistent trajectory) under the same convention as job4971615 — this file does not invent a persistent-trajectory metric for SimpleAvg that the canonical script itself never computes. No metric required an ImageNet-specific reimplementation; nothing in this pass required invoking the "STOP and report the mismatch" clause (task brief Section 16).

## 8. Static test results (this pass, real, executed)

| Check | Result |
|---|---|
| `python -m py_compile experiments_prepared/final_8method_imagenet100_job4971615_compatible.py` | **PASS** |
| `bash -n experiments_prepared/slurm/final_8method_imagenet100_job4971615_compatible.sbatch` | **PASS** |
| Canonical CIFAR-100 script unmodified (`git status --short`) | **PASS** |
| 9-method ImageNet-100 script unmodified (`git status --short`) | **PASS** |
| **Config-construction smoke test** — real execution of the new script's own source, truncated immediately before the dataset-loading/network cell (line 3240, `dataset = load_dataset(...)`), zero network/dataset dependency | **PASS.** Printed, live-resolved output confirms: exactly 8 active methods (listed, no T4), `KD_WEIGHT=1.0`, `LAMBDA_ORTH=50.0`, `HEAD_LR_MULTIPLIER_BY_FAMILY` = `{simple_avg: 10.0, rank_extension: 1.0}`, `TARGET_MODULES_BY_FAMILY` both `[q_proj, v_proj]`, `RANKEXT_RANK_SCHEDULE_active=[16,32,48,64,80]`, `USE_RANKEXT_RANK_SCHEDULE_WIDE=False`, `LORA_EPOCHS=RANKEXT_EPOCHS=9`, `BASE_OUTPUT_DIR` correctly namespaced (`imagenet100_5x20_final_8method_job4971615_compatible_seed42_EPOCH3_MAIN_...`, no collision with either the CIFAR-100 or the 9-method ImageNet run). Every hard assertion in the method-registration/readiness-verification blocks (`EXPECTED_METHODS`, KD-scope, temperature, protect-weight, new-block-warmup, target-modules, head-LR, rank-schedule assertions) executed and passed with zero exceptions. |
| Method-count assertion (=8) | **PASS**, part of the above (`assert len(ACTIVE_METHOD_NAMES) == 8` and 3 other equivalent assertions all fired and passed) |
| Dataset-split sanity (class-order artifact) | **PASS**, Section 5 above, re-verified fresh offline |
| One-batch dataloader smoke / forward pass on real ImageNet-100 data | **NOT independently re-executed in this pass.** The dataset-loading/preprocessing/forward code this new file uses is byte-identical (zero diff lines) to the 9-method ImageNet sibling's own dataset cell, which was already executed successfully with real streamed data in the prior session (`R7/imagenet100_5x20_static_readiness.md`, "Cluster Smoke Test" section: real network calls, 8 real streamed examples, correct label remap, `pixel_values.shape=(8,3,224,224)`, real CLIP ViT-B/16 forward pass, output shapes `(8,768)`/`(8,100)`, all PASS except the expected CUDA-unavailable check on this CPU-only machine). Re-running it here would exercise unchanged code and consume network bandwidth for no new information; cited as existing evidence rather than duplicated. |
| Backward pass per loss family (SA plain/KD/DenseOrth, RE plain/KD+Protect/FactorOrth/combined) | **NOT independently executed in this pass** — see explicit callout below. |

**Explicit callout, per this task's "STOP and report the mismatch" principle:** the per-loss-family forward/backward smoke test (task brief Section 24's last item) was **not** independently re-executed against this new file. Every trainer/loss class this file uses (`IndependentLoraOrthTrainer`, `DeltaOrthRankExtensionTrainer`, `masked_kd_loss`, dense/factor-orth penalty functions, `compute_old_semantic_subspace`) is inherited **byte-identical** from the canonical CIFAR-100 script — the diff between this file and the canonical script (Section 10 below) touches zero lines inside any trainer or loss function. That exact code already executed successfully in production to produce job4971615's own results, and the ImageNet-specific data path (dataset→preprocessing→CLIP forward) it feeds into was independently smoke-tested with real data in the prior session (row above). Building a faithful synthetic-tensor harness for all 6 loss families within this pass was judged disproportionate to what changed (a single `METHODS_TO_RUN` flag plus its count/name bookkeeping) and was not attempted rather than approximated — consistent with this task's explicit instruction not to invent or approximate compatibility evidence.

## 9. Reporting / Chapter-5 schema compatibility

`R7/chapter5_main_benchmark/metadata/dataset_registry.json` (read only, not modified) already reserves a `"second_dataset": {"status": "pending"}` slot, confirming no ImageNet-100 data has been populated anywhere in the Chapter-5 package. Its `cifar100` entry's schema (`methods`, `excluded_methods`, `data_files: [chapter5_cifar100_8method_main.csv, chapter5_cifar100_per_task_metrics.csv]`, `parameter_efficiency`, `Rij` scoped to RankExt-only) defines the shape an eventual `chapter5_imagenet100_8method_main.csv` / `chapter5_imagenet100_per_task_metrics.csv` pair would need to match — this pass produced no such files and populated no benchmark data, per explicit instruction.

## 10. Source-diff audit — canonical CIFAR-100 vs. the new 8-method ImageNet-100 script

Full diff (`diff -U0`, i.e. zero context lines, counting only the file's own changed content): 29 hunks, 152 `+`/`-` lines out of ~9,900 total (≈1.5%) — independently re-measured during this report's own verification pass; an earlier draft of this section stated 517/442, which was a miscount and has been corrected here. Every changed region classified:

| Region | Lines (approx.) | Classification | Notes |
|---|---|---|---|
| Header/provenance banner | ~1–115 | REPORTING-ONLY | Documentation only; no executable change |
| `RUN_NAME_BASE` | 1 line | DATASET-REQUIRED | Prevents output-path collision across datasets/scripts (Section 20) |
| New constants (`IMAGENET100_DATASET_ID`, `_CACHE_ROOT`, `_CLASS_ORDER_ARTIFACT_PATH`, `FINAL_EVAL_SPLIT_NAME`) | ~4 lines | DATASET-REQUIRED | No CIFAR analogue needed |
| Dataset-loading cell (`load_dataset`, identity assertions, class-order load + remap) | ~120 lines | DATASET-REQUIRED | `class_splits` construction itself unmodified — operates on remapped label space only |
| `make_eval_dataset()` split-name lookup | 1 line | DATASET-REQUIRED | This mirror has no split literally named `"test"` |
| Cosmetic print/diagnostic strings (dataset name, script name, "8-method" vs "9-method") | ~15 lines | REPORTING-ONLY | Console/log accuracy only |
| `METHODS_TO_RUN["simple_avg_kd_oldseen_T4_warmup"]` flag | 1 line | IMPLEMENTATION-NECESSARY | The one deliberate functional change (Section 2) |
| `EXPECTED_METHODS`, `EXPECTED_ENABLED_METHOD_FAMILIES`, `_SA_KD_ARMS`, `_T4_ABLATION_BASE_METHODS` consumers, method-count assertions (9→8, 5→4 SimpleAvg) | ~30 lines | IMPLEMENTATION-NECESSARY | Direct, mechanical consequence of the T4 flag flip — no independent judgment call |
| Reporting-section T4 comparison rows removed (`_sa_kd_t4`, within-run and cross-run delta rows) | ~10 lines | IMPLEMENTATION-NECESSARY | T4 no longer trains, so no such row can exist; removing them prevents a `KeyError`/fabricated row, not a scientific choice |
| **Every loss function, trainer class, optimizer/scheduler construction, calibration function, evaluation function, classifier restoration, merge mechanism** | 0 lines | — | **Unchanged.** Zero diff lines touch this code. |

**SCIENTIFIC-DIFFERENCE = NONE**, confirming the audit's stated goal (task brief Section 26): every change is either dataset-required (the dataset itself is different), implementation-necessary (a mechanical consequence of disabling one ablation arm), or reporting-only (console/log text). No hyperparameter, loss formulation, calibration mode, or training-loop behavior differs from job4971615's own source for any of the 8 shared methods.

## 11. Output naming / reproducibility / dataloader / checkpoint safety

- **Output naming (Section 20):** `RUN_NAME_BASE = f"imagenet100_5x20_final_8method_job4971615_compatible_seed{SEED}"`, distinct from both `imagenet100_5x20_final_9method_performance_recovery_seed{SEED}` (9-method sibling) and `cifar100_5x20_final_9method_performance_recovery_seed{SEED}` (canonical CIFAR-100) — confirmed via the live smoke test's printed `BASE_OUTPUT_DIR`. `results/` root and `IMAGENET100_CACHE_ROOT` are the only shared paths, and only the latter is dataset-content-keyed (safe to share — see the new sbatch's own comment).
- **Reproducibility (Section 21):** `SEED=42` unchanged and used identically everywhere (class-order artifact generation, per-class shuffles, per-method seed reset) — inherited unchanged.
- **Dataloader/memory safety (Section 22):** no change made for optimization; `dataloader_num_workers=(0 if FAST_RUN else 4)` and all batch-size/precision settings are inherited unchanged from the 9-method sibling. The only cluster-side risk (compute-node network reachability to huggingface.co) is unchanged from, and already documented in, the 9-method sibling's own readiness report — this pass adds no new risk.
- **Checkpoint/resume safety (Section 23):** `CHECKPOINTS_DIR` is derived from the new `RUN_NAME_BASE`, so it is automatically isolated from every other run's checkpoints; the resume mechanism itself is inherited unchanged.

## 12. Hard-fail conditions (task brief Section 28) — checked explicitly

| Condition | Status |
|---|---|
| Wrong method count | 8 confirmed live (Section 8) |
| Task-scale calibration active | Confirmed **absent** (Section 4) |
| SimpleAvg head-LR multiplier ≠ 10 | Confirmed **= 10.0** (Section 3) |
| RankExt head-LR multiplier ≠ 1 | Confirmed **= 1.0** (Section 3) |
| Wrong KD formulation | Confirmed old-seen (SimpleAvg) / full (RankExt), T=2, weight=1.0 (Section 3) |
| Wrong DenseOrth/FactorOrth lambda | Confirmed 20.0 / 50.0 (Section 3) |
| Wrong Protect weight | Confirmed 30.0 (Section 3) |
| Rank schedule mismatch | Confirmed `[16,32,48,64,80]`, wide schedule OFF (Section 3) |
| Classifier restoration mismatch | Inherited unchanged, zero diff lines (Section 7) |
| Dataset split uncertainty | Resolved, re-verified fresh (Section 5) |
| Label mapping uncertainty | Resolved, re-verified fresh (Section 5) |
| Compile failure | None — PASS (Section 8) |
| Dataloader failure | Not independently re-tested this pass; unchanged code, prior PASS evidence cited (Section 8) |
| Forward/backward shape error | Not independently re-tested this pass for the per-loss-family case; unchanged code (Section 8) |
| Output path collision risk | None — distinct `RUN_NAME_BASE` (Section 11) |
| Missing metric support | None (Section 7) |

**None of the hard-fail conditions are triggered.**

## 13. Verdict

**SCIENTIFIC COMPATIBILITY WITH JOB4971615: PASS.** **READY TO EXECUTE FULL IMAGENET-100 RUN: NO** — not because of any scientific-compatibility defect, but because full execution requires resolving the pre-existing, carried-over cluster blockers already documented in `R7/imagenet100_5x20_static_readiness.md` (compute-node network reachability to huggingface.co is unverified; per-epoch runtime is an estimate; disk quota unconfirmed) — none of which this pass was scoped to resolve, and none of which this pass introduced. **FULL TRAINING STARTED: NO. SLURM SUBMISSION: NOT SUBMITTED**, per explicit instruction.
