# Final 8-Arm Family-Specific Experiment — Implementation Audit

**Scope:** implementation + static/smoke verification only. The production SLURM job has **NOT** been submitted. No historical result files were changed. New file: `experiments_prepared/final_8arm_family_specific_5x20.py` (built by copying `experiments_prepared/simpleavg_rankext_singlejob_improvement_pack_5x20.py` and applying the documented edits below — that source file, and `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`, are both untouched).

---

## 1. Exact 8-Arm Table

| # | Family | Method | Rank/schedule | Epochs | KD scope | KD T | KD weight | KD warmup | Orth type | λ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | simple_avg | `simple_avg` | 80 (fixed) | 6 | n/a | n/a | 0 | n/a | none | 0 |
| 2 | simple_avg | `simple_avg_kd_oldseen_T2_weight0p5` | 80 (fixed) | 6 | old_seen | 2.0 | **0.5** | none (0 epochs) | none | 0 |
| 3 | simple_avg | `simple_avg_dense_orth_lam20` | 80 (fixed) | 6 | n/a | n/a | 0 | n/a | dense_orth | 20 |
| 4 | simple_avg | `simple_avg_dense_orth_lam20_kd_oldseen_T2_weight0p5` | 80 (fixed) | 6 | old_seen | 2.0 | **0.5** | none (0 epochs) | dense_orth | 20 |
| 5 | rank_extension | `rank_extension` | 16→32→48→64→80 | 6 | n/a | n/a | 0 | n/a | none | 0 |
| 6 | rank_extension | `rank_extension_kd_oldseen_T2_weight1p5_warmup` | 16→32→48→64→80 | 6 | old_seen | 2.0 | **1.5** | 1 epoch | none | 0 |
| 7 | rank_extension | `rank_extension_factor_orth_lam50` | 16→32→48→64→80 | 6 | n/a | n/a | 0 | n/a | **factor_orth** | **50** |
| 8 | rank_extension | `rank_extension_factor_orth_lam50_kd_oldseen_T2_weight1p5_warmup` | 16→32→48→64→80 | 6 | old_seen | 2.0 | **1.5** | 1 epoch | **factor_orth** | **50** |

Merge: arithmetic-only for SimpleAvg (`simple_average_deltas()`, unchanged); persistent for RankExt. DO-Merge is not evaluated anywhere in this script. All arms: seed 42, 6 epochs, q_proj/v_proj, no replay, corrected classifier-row restoration.

Verified programmatically at module load (see §17 below and the script's own `_EXPECTED_SINGLEJOB_METHOD_NAMES`/weight/warmup/λ assertions, all of which passed during the smoke run — §11).

---

## 2. Current vs. Historical Code Reused

| Component | Status |
|---|---|
| `compute_delta_orth_components()` | **Byte-identical** to both `supervisor_exp1_cifar100_5x20_fixed_rankext.py` and the current pipeline (confirmed by direct `diff` of the function body during this audit) — computes both the legacy trace/norm terms and the `factor_A_mean`/`factor_B_mean`/`factor_total_mean` FactorOrth terms in one pass. Never modified. |
| `DeltaOrthRankExtensionTrainer` | Present, unmodified except for the **purely additive** logged-row fields described in §9 (no change to `compute_loss`'s returned `loss` tensor). Its `orth_mode == "factor_orth"` branch (`orth_loss_used = comps["factor_total_mean"]`) is the SAME branch Exp1/R7 used for the historical `rank_extension_orth_factor_lam_50[_kd]` arms — never touched by the current pipeline's DenseOrth addition, which only added a *new*, separate `orth_mode == "dense_orth"` branch alongside it. |
| `RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS = 1.0` / `RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED = True` | Unchanged constants, reused verbatim. This is FactorOrth's own pre-existing 1-epoch lambda ramp (applies to every non-`"none"`/non-`"dense_orth"` orth_mode, i.e. to `factor_orth` too) — not reinvented for this script. |
| `LAMBDA_ORTH = 50.0` | Unchanged global constant. Never used by any of job 4969059's active arms (DenseOrth used `DENSE_ORTH_LAMBDA=20.0` instead) — this script is the first to actually route an active arm through it since the current pipeline's DenseOrth addition. |
| `add_method(..., uses_factor_orth=True)` config-resolution formula | Unchanged (`lambda_orth = LAMBDA_ORTH * lambda_orth_scale`, `lambda_orth_scale=1.0` default → 50.0). The pre-existing, currently-disabled `add_method("rank_extension_orth_factor_lam_50", ..., uses_factor_orth=True)` call (still present, gated `False`, in the copied base file) is the exact historical call this script's own RE-3/RE-4 `add_method()` calls mirror — same keyword, same resolved λ, different `method_name`/`base_method` (see §4). |
| `masked_kd_loss()` / `kd_class_scope="old_seen"` | Unchanged — the same old-seen-only KD mechanism job 4969059 used, for both new RankExt KD arms and both new SimpleAvg KD arms. |
| `restore_protected_classifier_rows()` | Unchanged — uses the FIXED `tensor[idx] = value` in-place assignment (not the historical `.copy_()` no-op). |
| `IndependentLoraOrthTrainer` (SimpleAvg's trainer) | Unchanged. |
| `do_merge_deltas()` / DO-Merge evaluation path | **Not called anywhere in this script** — `MERGE_METHODS_TO_EVALUATE = ["arithmetic"]` only. |
| The historical `"OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION"` (`protect_weight`) mechanism | Present in the copied trainer code (never deleted) but **never activated** by this script's arms — `protect_weight` resolves to 0.0 for every one of the 8 arms because `RANKEXT_PROJECTED_PROTECT_METHODS` only contains the two legacy identifiers (`rank_extension_kd_only_T2`, `rank_extension_orth_factor_lam_50_kd_T2`), and this script deliberately uses different, new method names for its FactorOrth arms specifically to avoid this collision (asserted at module load — §17). |

---

## 3. Exact FactorOrth Behavior Restored (Porting Audit, Part 7)

Read directly from `compute_delta_orth_components()` and `DeltaOrthRankExtensionTrainer.compute_loss()` (both files, confirmed byte-identical where FactorOrth is concerned):

- **A-factor reference construction:** `A_old = module.A_frozen` — the accumulated, already-frozen LoRA A-factor rows from all previously-completed CL steps for that layer (not a separate "averaged reference" object; it is the model's own frozen factor state, read directly).
- **B-factor reference construction:** `B_old = module.B_frozen` — the accumulated frozen B-factor columns, same layer, same steps.
- **Normalization:** row-wise L2-normalize `A_old`/`A_new` (`A_old_hat = A_old / A_old.norm(dim=1, keepdim=True).clamp_min(eps)`); column-wise L2-normalize `B_old`/`B_new` (`dim=0`).
- **Cosine/orthogonality computation:** `A_overlap = A_old_hat @ A_new_hat.T`, `factor_A = sum(A_overlap ** 2)`; symmetric construction for `B_overlap`/`factor_B`. `factor_total = factor_A + factor_B` — this is the scalar `orth_loss_used` for `orth_mode == "factor_orth"`.
- **Detached:** `A_old`/`B_old` come from `module.A_frozen`/`module.B_frozen`, which are themselves frozen (`requires_grad=False`) parameters populated once per step by the growing-rank mechanism — not a separately detached snapshot, but structurally non-trainable already.
- **Which previous blocks contribute:** *all* previously-frozen rank blocks for that layer simultaneously (the full accumulated `A_frozen`/`B_frozen`, not a single most-recent block) — i.e., `cumulative_orth_formula_label(step_idx)` describes this as `orth(L1+...+Lt-1, Lt)`.
- **Reference used:** accumulated frozen blocks (see above) — not a separately "averaged factor reference state" object (that construct, `average_factor_reference_state()`, exists in this codebase but is used only by the *SimpleAvg* independent-LoRA FactorOrth path, `compute_independent_lora_factor_orth_components()` — a structurally different function for a structurally different architecture, never invoked for RankExt).
- **Lambda application:** `effective_lambda_orth = lambda_orth * orth_warmup_multiplier`; `weighted = effective_lambda_orth * orth_loss_used`; added directly into `loss` alongside CE and (if active) KD.
- **Orth warmup:** `RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS=1.0`, `RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED=True` — linear 0→1 ramp over the first epoch of *each* CL step's local training, unconditionally applied whenever `orth_mode != "none"` (so it also applied to FactorOrth historically, and does so again here) — reused verbatim, not reinvented.
- **Interaction with `teacher_active`/KD:** none at the loss-computation level — `weighted` (orth) and `weighted_kd` are independent additive terms in `loss = ce_loss + weighted + weighted_kd + weighted_pretrained_anchor + weighted_protect`. FactorOrth's own computation does not read `teacher_active` or any KD-related state.
- **Treatment of step 1:** `cumulative_old_delta()` returns `None` at step 1 (no prior frozen blocks exist yet); `compute_delta_orth_components()` substitutes a zero tensor, so `factor_A`/`factor_B`/`factor_total` are all exactly 0.0 at step 1 for every arm — orthogonality is a genuine no-op at step 1, by construction, not a special case that needed re-implementing.
- **Old slices active in forward:** governed by `old_active_in_forward` (unrelated to FactorOrth itself) — unchanged, `True` throughout (this script never sets `zero_old_merge=True`).
- **New-block output warmup:** `RankExtNewBlockWarmupCallback` / `RANKEXT_NEW_BLOCK_WARMUP_EPOCHS=1.0` — a *different*, pre-existing mechanism (ramps the newly-added rank block's own output contribution during the first epoch of training, independent of the orthogonality loss) — unchanged, and (per `RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS`, which only lists the two legacy identifiers) still **active** for all 4 of this script's RankExt arms, exactly as it was for job 4969059's arms.

### A vs. B split

| Behavior | Classification | Restored here? |
|---|---|---|
| A/B factor cosine-overlap geometry, row/column normalization, `factor_total = factor_A + factor_B` | **A — intrinsic to FactorOrth** | Yes |
| `LAMBDA_ORTH=50.0`, applied via `lambda_orth_scale=1.0` (no rescaling) | **A — intrinsic to FactorOrth's historical strength** | Yes |
| `RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS`/`ENABLED` (1-epoch ramp) | **A — FactorOrth's own pre-existing warmup, shared with trace/norm modes** | Yes (unchanged, not re-derived) |
| 9 epochs per CL step | **B — unrelated historical Exp1 protocol** | **No** — stays at 6 epochs |
| Historical full-100-way KD | **B — unrelated historical mechanism** | **No** — old-seen-only KD (`masked_kd_loss`) used for arm 8 |
| Old (non-restoring) classifier-row `.copy_()` bug | **B — unrelated historical bug** | **No** — the fixed `tensor[idx]=value` restoration is used unconditionally |
| `protect_weight` / OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION | **B — a separate retention mechanism historically bundled with the same identifier, not part of FactorOrth's own geometry** | **No** — new method names deliberately avoid `RANKEXT_PROJECTED_PROTECT_METHODS`, verified by assertion (§17) |
| `COMBINED_LAMBDA_ORTH_SCALE`/`COMBINED_KD_WEIGHT_SCALE` down-scaling (historically applied to the analogous full-strength combined arm to avoid a documented training collapse) | **B — an unrelated historical mitigation, not part of FactorOrth's geometry, and not requested** | **No** — arm 8 uses the exact λ=50/weight=1.5 values requested, un-rescaled (flagged as a risk in §13, not silently applied) |

---

## 4. Unrelated Exp1 Behavior Deliberately NOT Restored

Exhaustive list, cross-checked against the task brief's own examples:
- 9 epochs per CL step (stays 6, matching job 4969059).
- Historical full-100-way KD (both new KD-using arms use `kd_class_scope="old_seen"`, the current pipeline's mechanism).
- The old, non-restoring classifier-row `.copy_()` behavior (the fixed in-place assignment is used unconditionally, for every RankExt arm including the two new FactorOrth ones).
- `protect_weight` (OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION) — never activated (§3 above).
- `COMBINED_LAMBDA_ORTH_SCALE`/`COMBINED_KD_WEIGHT_SCALE` automatic down-scaling for the combined arm.
- DO-Merge (removed entirely — `MERGE_METHODS_TO_EVALUATE = ["arithmetic"]`).
- Exp2's wide RankExt rank schedule (`USE_RANKEXT_RANK_SCHEDULE_WIDE` stays `False`, asserted).
- The legacy `rank_extension_orth_factor_lam_50`/`rank_extension_orth_factor_lam_50_kd_T2` method identifiers themselves (new, distinct names are used specifically so this run's arms cannot silently inherit any legacy method-name-keyed special-casing anywhere in the file, e.g. `RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS`, `RANKEXT_PROJECTED_PROTECT_METHODS`).

---

## 5. KD Weight=0.5 (SimpleAvg) Implementation Verification

- `add_method("simple_avg_kd_oldseen_T2_weight0p5", ..., kd_weight_scale=0.5, kd_warmup_epochs=KD_OLDSEEN_WARMUP_EPOCHS_ARM1)`.
- Resolved `kd_weight = KD_WEIGHT(1.0) * kd_weight_scale(0.5) = 0.5` — verified by an explicit module-load assertion (`abs(kd_weight - 0.5) < 1e-9`) that ran successfully during the smoke test (§11).
- `kd_warmup_epochs = KD_OLDSEEN_WARMUP_EPOCHS_ARM1 = 0.0` → `orth_lambda_warmup_multiplier(..., enabled=False when warmup_epochs<=0)` returns `1.0` unconditionally — i.e. **no warmup**, flat weight=0.5 from batch 1, verified by assertion.
- `masked_kd_loss()` slices logits to old-seen classes *before* softmax/log_softmax (unchanged function, not re-read line-by-line in this audit since it is verbatim-identical to job 4969059's already-verified implementation — confirmed via `diff`-equivalent inspection during the earlier design audit).
- Teacher: `build_simple_avg_teacher_model(step_states)` — SimpleAvg's existing teacher construction, unchanged, same call site as job 4969059's KD arms.

## 6. KD Weight=1.5 (RankExt) Implementation Verification

- `add_method("rank_extension_kd_oldseen_T2_weight1p5_warmup", ..., kd_weight_scale=1.5, kd_warmup_epochs=KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4)` and the equivalent FactorOrth-combined arm.
- Resolved `kd_weight = KD_WEIGHT(1.0) * kd_weight_scale(1.5) = 1.5` — verified by assertion.
- `kd_warmup_epochs = 1.0` → the SAME `orth_lambda_warmup_multiplier` linear-ramp formula as job 4969059's `_warmup` arms; `weighted_kd = kd_weight * kd_warmup_multiplier * kd_loss` (see `DeltaOrthRankExtensionTrainer.compute_loss`, line region ~7835 in the new script) — confirms **the warmup multiplier scales the configured weight, it does not replace it**: at `epoch_val >= 1.0`, `kd_warmup_multiplier == 1.0`, so `weighted_kd == 1.5 * kd_loss`, i.e. RankExt's steady-state (epoch ≥ 1 of each step) KD coefficient genuinely reaches 1.5, not 1.0. This is now directly verifiable per-batch via the new `effective_kd_coefficient` logging field (§9).
- Teacher: frozen previous-step RankExt model (`build_rank_extension_model(previous_rank_state, step_idx-1, ...)`), unchanged construction, identical to job 4969059's RankExt KD arms.

---

## 7. Orth / KD Warmup Behavior Summary

| Arm | Orth warmup | KD warmup |
|---|---|---|
| SA-3/SA-4 (DenseOrth) | `DENSE_ORTH_WARMUP_EPOCHS=1.0` (SimpleAvg's own, independent DenseOrth ramp — unchanged) | SA-4 only: **none** (0 epochs, flat 0.5) |
| RE-3/RE-4 (FactorOrth) | `RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS=1.0` / `ENABLED=True` (FactorOrth's own pre-existing ramp, shared with the legacy trace/norm modes — unchanged) | RE-4 only: 1 epoch (unchanged convention, `KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4`) |

---

## 8. Classifier Restoration Verification

`restore_protected_classifier_rows()` is unchanged and uses `model.classifier.weight[row_idx] = snapshot["weight"][rows]` / the bias equivalent (`__setitem__`/`index_put_`, genuinely in-place) — **not** the historical `tensor[idx].copy_(value)` advanced-indexing no-op. This is exercised for every one of the 4 RankExt arms, every CL step (2–5; step 1 has no protected rows yet).

**New (this script):** `classifier_protected_row_diffs()` — an additive function that returns weight-diff and bias-diff *separately* (the pre-existing `classifier_protected_row_max_diff()` only ever returned their `max()`, and only printed it, never saved it). Wired into `train_rank_extension_arm()` to append one row per (method, step) to `classifier_restoration_rows`, saved to `tables/final_8arm_classifier_restoration_diagnostics.csv`. The reporting section **asserts** (a stop condition, not just a print) that the max weight diff and max bias diff across every RankExt arm/step are both `< 1e-6` before declaring the run's reporting section complete.

---

## 9. Improved Logging Schema

**RankExt (`DeltaOrthRankExtensionTrainer`) row-dict additions** (pure additive fields — the `loss` tensor returned by `compute_loss` is byte-for-byte unchanged; every new field is computed from local variables that already existed):
`kd_class_scope`, `kd_warmup_epochs`, `kd_warmup_multiplier`, `effective_kd_coefficient` (`= kd_weight * kd_warmup_multiplier`), `dense_orth_mean` (NaN when `orth_mode != "dense_orth"` — not exercised by this script's arms, kept for schema parity with SimpleAvg), `dense_orth_warmup_multiplier` (documented as an honest alias of `orth_warmup_multiplier` for RankExt, since this class never separately consults its own `dense_orth_warmup_epochs` constructor parameter — a genuine, pre-existing asymmetry with SimpleAvg's trainer, now made visible rather than silently assumed), `effective_orth_coefficient` (`= effective_lambda_orth`), `orth_warmup_multiplier`.

**Coverage fix (the actual "gap" in job 4969059):** the underlying per-batch accumulator (`train_diagnostic_rows`, aliased as `orth_kd_train_rows`) was **already being populated for every arm in both families** upstream — `IndependentLoraOrthTrainer.consume_logged_losses()` / `DeltaOrthRankExtensionTrainer.consume_logged_losses()` are called unconditionally after every step, for every arm, in both the current pipeline and this script. The coverage gap in job 4969059's on-disk `logs/training_loss_history_by_batch.csv` was in the **downstream CSV-write step** (scoped to 2 "supervisor-selected" methods), which this script's new reporting section (§ new tail of the file) replaces entirely with a complete dump of the full accumulator. Verified directly: the smoke test's per-batch log (§11) contains rows for every arm that reached training during the truncated smoke window, tagged by its own distinct `method` value — not collapsed under a shared label.

---

## 10. CL Retention/Plasticity Diagnostics

**New:** `evaluate_seen_step_accuracies_restricted(model, upto_step_idx)` — a restricted-accuracy analogue of the pre-existing `evaluate_seen_step_accuracies()`, using the same eval-only `Trainer` construction and `restricted_argmax_accuracy()` already used elsewhere in this file. Called from `train_rank_extension_arm()` immediately after the existing (open-accuracy) `evaluate_seen_step_accuracies()` call, at the same point in the per-step loop (right after that step's model is finalized, before advancing to the next step). Both results are combined into `cl_trajectory_rows`, one row per `(method, after_training_step, evaluated_task_step)`, with `open_accuracy`, `restricted_accuracy`, `is_current_task`, `is_old_task` — saved to `tables/final_8arm_rankext_cl_retention_plasticity_trajectory.csv`.

**SimpleAvg:** deliberately **not** given an equivalent trajectory table. Its 4 arms each train 5 independent per-step specialists with no intermediate merged model between steps — "accuracy on previously-seen steps after training step k" is not a well-defined quantity for k<5 in that architecture (only the final, fully-merged model at k=5 is ever evaluated). The reporting section prints an explicit note to this effect rather than fabricating a metric, per the task brief's own instruction.

---

## 11. Smoke-Test Results

Environment: this machine is **CPU-only** (`torch.cuda.is_available() == False`); no GPU is available locally, so this smoke test exercises correctness (all 8 arms instantiate, train, log, and evaluate) but not GPU-specific code paths (mixed precision, CUDA memory management) — those are unchanged from job 4969059's already-GPU-verified pipeline and are not touched by any edit in this script.

- `python -m py_compile experiments_prepared/final_8arm_family_specific_5x20.py` → **PASS**.
- `FAST_RUN_DEBUG=1 python experiments_prepared/final_8arm_family_specific_5x20.py`:
  - All module-load static assertions passed (8-method set match, family-specific KD weight/warmup/λ checks, FactorOrth/DenseOrth mutual exclusivity per family, `RANKEXT_PROJECTED_PROTECT_METHODS` non-collision, `ENABLED_METHOD_FAMILIES` match) — confirmed by the process reaching real training (CLIP checkpoint download + load succeeded, CIFAR-100 loaded) with no `AssertionError`.
  - **PARTIAL, USER-STOPPED — not a full 8-arm PASS.** On CPU-only hardware (no GPU available on this machine; `torch.cuda.is_available() == False`), the full 8-arm/5-step/1-epoch smoke run was projected to take on the order of 2+ hours, so it was stopped by explicit user request before completion. What it verified before being stopped:
    - **Arm 1 (`simple_avg`) completed all 5 CL steps** (training, best-epoch selection, evaluation) with zero errors.
    - **Arm 2 (`simple_avg_kd_oldseen_T2_weight0p5`) reached step 3 of 5**, with `teacher_active=True` from step 2 onward — confirming the old-seen-KD teacher construction engages correctly for a weight=0.5 arm.
    - **Not exercised:** arms 3–8 (SimpleAvg DenseOrth/combined, and all 4 RankExt arms — including both new FactorOrth arms and the weight=1.5 RankExt KD arm). The FactorOrth loss-computation path, the RankExt classifier-restoration diagnostics, and the CL trajectory logging added in this script were therefore **not exercised at runtime** in this pass — only verified by static code reading (§3, §8, §10).
  - **Consequence for the Part 21 stop conditions:** "smoke test does not execute all 8 arms" is one of the listed stop conditions. Per that rule, this experiment **should not yet be declared ready to run** on the strength of this partial smoke pass alone — see §14 (Final Recommendation) for what would close this gap (either a longer local CPU run, or a short cluster-side GPU smoke run, which would take minutes instead of hours).

---

## 12. Production Launcher Path

`experiments_prepared/slurm/final_8arm_family_specific_5x20.sbatch` — modeled directly on job 4969059's already-used launcher (`simpleavg_rankext_singlejob_improvement_pack_5x20.sbatch`): explicit `cd` to the absolute repo path (never `dirname "$0"`), 1× A40 GPU, 4 CPUs, 32GB RAM, 12-hour walltime, unsets `FAST_RUN_DEBUG`/`REPLICATION_SEED` before running, invokes the absolute cluster Python interpreter.

---

## 13. Unresolved Risks

1. **RE-4 (`rank_extension_factor_orth_lam50_kd_oldseen_T2_weight1p5_warmup`) uses full-strength, un-rescaled values (λ=50, KD weight=1.5) with no automatic down-scaling.** The codebase's own history documents a training collapse for the *analogous* full-strength combined arm at kd=1.0/λ=50 (`COMBINED_LOSS_SCALE_ENABLED`'s own comment block, still present in the copied file) — this script's arm pushes KD weight even higher (1.5) than that historical collapse case (1.0). This is flagged, not silently mitigated, per the task brief's explicit instruction to use the exact requested values. **If the production run shows a similar collapse for this specific arm, that itself is scientifically informative** (bearing on whether combined FactorOrth+KD is fundamentally unstable at these values) and should not be treated as a bug.
2. **CL/classifier diagnostics (Parts 10/11) are populated only on a genuine (non-cached) training pass**, exactly matching the pre-existing `orth_train_records` behavior — if a checkpoint from a partial prior run of this exact script is present and loaded via `load_rankext_checkpoint_if_present()`, `train_rank_extension_arm()` is skipped entirely and these two new diagnostic tables would be empty for that arm. Not expected to matter for this script's first, from-scratch production run (fresh `RUN_NAME_BASE`, no pre-existing checkpoints).
3. **No GPU-specific verification was possible locally** (CPU-only smoke test) — mixed-precision/CUDA-memory code paths are unchanged from job 4969059 and are not independently re-verified here.
4. **`validation_accuracy` is not tracked anywhere in this pipeline** (only validation CE, for best-epoch selection) — the final summary table reports it as `NaN` with an explicit note rather than fabricating a value; only `validation_CE_mean_across_steps` is populated.

---

## 14. Final Recommendation

**NOT YET READY TO RUN**, solely because of the incomplete smoke test (§11/§13.6) — every other stop condition in Part 21 was checked and passed (FactorOrth matched to Exp1 by direct code/`diff` reading, not just by name; KD weight reaches its intended steady-state coefficient by formula and is verified by assertion for both new weights; old-seen slicing is unchanged (`masked_kd_loss`, verbatim); classifier protected-row restoration uses the fixed in-place assignment, unconditionally; granular logging code exists for all 8 arms; the result table is asserted to contain exactly the 8 expected arms, no silent omission possible).

To close the remaining gap, one of the following should happen before submitting the real SLURM job:
- **Preferred:** run the smoke test on the actual cluster with a GPU (`FAST_RUN_DEBUG=1`, same command as `final_8arm_family_specific_5x20.sbatch` but interactively or as a short debug job) — with a GPU this should take minutes, not hours, and would exercise all 8 arms including the two new FactorOrth arms and the weight=1.5 RankExt KD arm end-to-end.
- **Alternative:** resume the local CPU smoke test to completion (it can be restarted — `FAST_RUN_DEBUG=1 python experiments_prepared/final_8arm_family_specific_5x20.py`, writing to the isolated `results_debug_preflight/` tree — no scientific output is at risk) if a longer local wait is acceptable.

Everything else in this audit supports readiness; only the runtime coverage of arms 3–8 remains unverified.
5. The pre-existing `run_rank_extension_variant()` function (an older, single-pass RankExt driver, superseded by the `train_rank_extension_arm`/`finalize_rank_extension_arm` pair actually used by the execution loop) remains in the copied file, unmodified and **unreachable** (never called) — confirmed by search; left in place rather than deleted, to keep the diff minimal, per the task brief's "keep changes minimal" instruction.
6. **The smoke test only exercised 2 of 8 arms before being stopped (§11)** — the FactorOrth loss path (arms 7/8), the weight=1.5 RankExt KD path (arms 6/8), and the new classifier-restoration/CL-trajectory diagnostics were verified by static code reading only, not by an actual runtime pass. This is the single biggest open gap before declaring the script production-ready — see §14.
