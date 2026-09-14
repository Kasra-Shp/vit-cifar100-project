# R8 Preparation: Both-Families Regularizer Repair (SimpleAvg + RankExt)

**Status: PREPARED, NOT LAUNCHED.** This document describes
`experiments_prepared/supervisor_regularizer_repair_both_families_r8.py`, a new, wholly
separate, additive script. R7 and its historical executable script
(`experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`) are untouched — verified
in Section 14.

---

## 1. Motivation from R7

R7 (`thesis_agent/reports/r7_simpleavg_kd_factororth_forensic_analysis.md`) found that, in
SimpleAvg: (a) full-100-way KD costs ~12 accuracy points, concentrated almost entirely in an
open-argmax "antiquity bias" rather than a loss of each step's own discriminative ability; and
(b) factor-space FactorOrth (elementwise mean of independently-trained, non-identifiable LoRA
A/B factors) massively dominates early training (λ=50 → ~144× CE) before decaying to ≈0, costing
convergence speed more than final accuracy. No implementation bug was found; the phenomenon was
classified as an architecture–regularizer mismatch, reinforced by RankExt's very different,
well-posed FactorOrth reference (literal frozen persistent blocks in one shared basis) — which is
exactly why the same regularizers *help* RankExt (Δ up to +35 pp) while hurting SimpleAvg.

This task's premise: fixing SimpleAvg alone is not a fair scientific test of "did the correction
work," because RankExt was never given the same corrected KD/orthogonality definitions to see how
*it* responds. R8 corrects both families with the identical conceptual fixes, so any future
result is a genuine before/after comparison, not a SimpleAvg-only patch.

## 2. Fairness rationale

Two independent risks if only SimpleAvg were corrected: (1) a result showing "corrected SimpleAvg
now matches plain SimpleAvg" would say nothing about whether the *correction* is sound or whether
SimpleAvg's baseline is just easy to match; (2) RankExt's historical FactorOrth/KD success could
not be distinguished from "RankExt would have succeeded with any reasonable regularizer" vs.
"RankExt specifically needs its own well-posed reference." Running the same correction on both
families answers both questions with one experiment. Per the request's fairness rule, every
*principle* (KD support, KD warmup, orth formula, orth warmup, target λ range, T, base KD weight,
diagnostic framework) is implemented as one shared function called identically by both families'
trainers (verified in Section 13); only genuinely architectural differences (teacher construction,
persistent-vs-independent adapter storage, merge mechanism, classifier handling) are allowed to
differ, and are listed explicitly in Section 9.

## 3. Corrected KD formulation

For CL step `t` (0-based `step_idx`, so step 1 = `step_idx=0`):

```
C_old        = classes introduced strictly before step t  (= [] at step_idx=0)
teacher_old  = teacher_logits[:, C_old]
student_old  = student_logits[:, C_old]
teacher_probs      = softmax(teacher_old / T)
student_log_probs  = log_softmax(student_old / T)
KD = KL(teacher_probs || student_probs) * T^2
kd_weight_effective = KD_BASE_WEIGHT * linear_warmup_multiplier(local_epoch, 1.0 epoch, enabled=True)
```

`T = 2.0`, `KD_BASE_WEIGHT = 1.0`, current-step-only training data, no replay — all unchanged from
R7. Implemented once, as `masked_kd_loss()` + `linear_warmup_multiplier()`
(`experiments_prepared/supervisor_regularizer_repair_both_families_r8.py`, Section 3), called
identically from `SimpleAvgCorrectedTrainer.compute_loss()` and
`RankExtCorrectedTrainer.compute_loss()`. Current- and future-class logits are never read by the
KD loss at all (not masked-to-zero — sliced out before softmax, so they cannot leak any gradient
or probability mass into the KD term); this is asserted at runtime by
`assert_kd_mask_excludes_current_and_future()`, exercised for every step by the self-test.

## 4. Corrected DenseOrth formulation

```
delta_current = scaling * B_current @ A_current                      (live, current step/block)
delta_i       = scaling * B_i @ A_i                                    (each PREVIOUS step/block, detached)
cos(delta_current, delta_i) = <delta_current, delta_i>_F / (||delta_current||_F ||delta_i||_F + eps)
penalty_l     = mean_i [ cos(delta_current, delta_i)^2 ]
effective_lambda = SELECTED_SHARED_LAMBDA * linear_warmup_multiplier(local_epoch, 1.0 epoch, enabled=True)
```

Implemented once as `dense_cosine_sq()` + `dense_orth_penalty()`, called identically by both
trainers. `dense_orth_penalty()` asserts every previous delta passed to it has `requires_grad ==
False` (raises otherwise) and never detaches the current delta, so gradient flow is enforced by
construction, not merely assumed. Reference is **individual previous deltas**, never a mean —
per the request's "why individual prior deltas" rationale, and directly avoiding R7's identified
factor-mean-of-non-identifiable-adapters problem.

## 5. SimpleAvg implementation

- Current delta: `scaling * B_current @ A_current` read live off the LoRA adapter currently being
  trained for this step (`SimpleAvgCorrectedTrainer.compute_loss()`).
- References: `previous_dense_deltas[module_name]` = **one dense delta per previous independent
  SimpleAvg step** (never averaged), produced by the same `extract_lora_state()` used to build the
  merge/teacher, and re-`.detach()`-ed defensively before being passed to `dense_orth_penalty()`.
- Neither A nor B factors are ever averaged in the corrected path — `average_factor_reference_state`
  (R7's mechanism) is not called anywhere in this file.
- KD teacher: unchanged family-specific construction — `build_simple_avg_teacher_model()` (frozen,
  eval-mode, dense-merged-so-far model + stitched classifier), reimplemented locally, functionally
  identical to R7's mechanism. Only the KD *loss* on top of this teacher is corrected (old-seen mask
  + warmup); the teacher itself is deliberately left as-is per the request's "KD teacher remains
  family-specific" rule.
- Merge (`simple_average_deltas` + `apply_deltas_to_base`, dense arithmetic mean + classifier
  row-stitching) and calibration (`confidence_weighted_regime_grouped`) are reimplemented
  identically to R7 — untouched design.

## 6. RankExt implementation

- New class `GrowingRankLoRALinearDenseOrth` (NOT a modification of R7's `GrowingRankLoRALinear`):
  keeps each previous step's incremental rank block **separately** in `self.frozen_blocks` (a
  list of individually frozen `(A_i, B_i)` pairs), instead of R7's design of collapsing all history
  into one concatenated `A_frozen`/`B_frozen` slice — this is required so DenseOrth can reference
  each individual previous block, per the request.
- `current_new_delta()` = `scaling * B_new @ A_new` (live, current step's new block only).
- `previous_block_deltas()` = `[scaling * B_i @ A_i for each frozen (A_i, B_i)]`, each `.detach()`-ed.
- Rank increments derived from the same historical schedule `[16, 32, 48, 64, 80]`:
  `rankext_new_rank_for_step()` gives `[16, 16, 16, 16, 16]` — verified by the self-test.
- Scaling is the constant `RANKEXT_ALPHA_PER_RANK = 2.0` for every block regardless of when it was
  added (matches R7's actual behavior: `scaling = alpha_per_rank * total_rank / total_rank =
  alpha_per_rank`, i.e. never rank-dependent).
- KD teacher: unchanged family-specific construction — `build_rankext_teacher_model()` is a
  `copy.deepcopy()` of the actual previous-step persistent RankExt model, frozen/eval-mode. This is
  architecturally different from SimpleAvg's dense-merge teacher **by design** (the request:
  "Do NOT artificially make teacher construction identical because the architectures differ").
- The historical `A_frozen`/`B_frozen` factor-overlap loss (`compute_delta_orth_components()` in
  R7) is not called anywhere in this file — DenseOrth replaces it entirely for the R8 methods.
- The pre-existing, unrelated `RANKEXT_NEW_BLOCK_WARMUP` mechanism (ramps the new block's own
  forward-pass contribution during its first epoch) is kept as-is (`_new_block_warmup_multiplier`
  in `GrowingRankLoRALinearDenseOrth.forward()`), per "match R7 otherwise" — it is independent of,
  and not conflated with, the new KD-weight/orth-lambda warmups.

## 7. Shared λ diagnostic

Implemented as `evaluate_lambda_candidates_on_batch()` (pure arithmetic: CE + raw-orth → per-λ
table) and `pick_shared_lambda()` (selects the smallest candidate in `{1, 5, 10, 50}` whose
`weighted_orth / CE` ratio lands in `[0.1, 1.0]` for **every** family simultaneously; if none does,
returns `None` plus an explicit conflict report — never silently falling back to a per-family λ).
Both functions are unit-tested with synthetic numbers in the self-test (no model/dataset needed for
the *decision logic*).

**UPDATE (diagnostic actually run — see `thesis_agent/reports/r8_shared_lambda_diagnostic.md` for
full detail):** `run_lambda_diagnostic()` has now been executed for real
(`--mode lambda_diagnostic`), using real CLIP-ViT-B/16 + CIFAR-100 (both already locally cached)
and a bounded amount of real training (15+10 CE-only optimizer steps per family to reach a
non-degenerate step-2 state, then 6 no-update measurement batches). Result:

- **CONFLICT — no shared λ found** among the tested candidates {1, 5, 10, 50}. At λ=50 (the
  largest candidate), `weighted_orth/CE` reaches only ≈9.6e-4 for SimpleAvg and ≈9.1e-5 for
  RankExt — both several orders of magnitude below the target [0.1, 1.0] range.
- Normalization verified correct (`scale_invariance_ok: {simple_avg: true, rank_extension: true}`)
  — the tiny ratios are a genuine measurement, not a normalization bug.
- The measurement is from an early-training regime (25 gradient steps) and likely *underestimates*
  the eventual, fully-trained `raw_orth` scale by roughly 7–27× (cross-referenced against R7's own
  fully-trained SimpleAvg delta-cosine data) — but even that correction would not bring any tested
  candidate into range.
- Per the standing instruction, this conflict was **reported, not resolved by silently picking a
  value** from this (loss-ratio) diagnostic.

**Superseded (see `thesis_agent/reports/r8_denseorth_formulation_and_gradient_audit.md` for the
full record):** a follow-up **gradient-ratio** diagnostic (`--mode gradient_diagnostic`) measured
`‖g_orth‖/‖g_ce‖` directly instead of loss magnitude, and found overlapping candidate ranges for
the two families ([17.4, 173.9] SimpleAvg, [6.4, 63.5] RankExt, overlap [17.4, 63.5]) — unlike the
loss-ratio criterion. `SELECTED_SHARED_LAMBDA = 20.0` was set from that measurement (the
conservative edge of the overlap), `--mode train` no longer refuses for a missing lambda, and the
previously-missing 8-method/5-step orchestration has since been implemented. **Full R8 training
has still not been launched.**

## 8. Warmup symmetry

One function, `linear_warmup_multiplier(local_epoch, warmup_epochs, enabled)`, used for all four
warmup call sites: SimpleAvg-KD, RankExt-KD, SimpleAvg-DenseOrth, RankExt-DenseOrth — all with
`warmup_epochs=1.0`, `enabled=True`. This explicitly removes R7's asymmetry (SimpleAvg+FO-alone had
no warmup at all; SimpleAvg+FO+KD had one via `teacher_active` gating; RankExt had it
unconditionally in both cases) — every R8 DenseOrth variant, in both families, now gets the same
one-epoch linear ramp regardless of whether KD is also active. Verified directly in the self-test
by source inspection (`inspect.getsource`) confirming both trainer classes call the identical
`ORTH_WARMUP_EPOCHS, ORTH_WARMUP_ENABLED` constants, not family-specific ones.

## 9. Exact 8 methods

| # | internal_name | family | KD | DenseOrth |
|---|---|---|---|---|
| 1 | `simple_avg` | simple_avg | no | no |
| 2 | `simple_avg_kd_oldseen_T2` | simple_avg | yes | no |
| 3 | `simple_avg_dense_orth` | simple_avg | no | yes |
| 4 | `simple_avg_dense_orth_kd_oldseen_T2` | simple_avg | yes | yes |
| 5 | `rank_extension` | rank_extension | no | no |
| 6 | `rank_extension_kd_oldseen_T2` | rank_extension | yes | no |
| 7 | `rank_extension_dense_orth` | rank_extension | no | yes |
| 8 | `rank_extension_dense_orth_kd_oldseen_T2` | rank_extension | yes | yes |

Methods 1 and 5 are plain controls, trained fresh by this file's own code — never by calling into
or reusing R7's artifacts.

## 10. Unchanged experimental invariants

Dataset=CIFAR-100, protocol=5×20, seed=42, epochs=9/step, SimpleAvg(rank=80, alpha=160,
scaling=2.0), RankExt(schedule=[16,32,48,64,80], scaling=2.0), targets=[q_proj, v_proj], optimizer=
AdamW, scheduler=cosine, per-family head-LR multiplier (10.0 / 1.0), calibration=
`confidence_weighted_regime_grouped` (both families), open/restricted evaluation definitions,
best-epoch selection (CE-only argmin), classifier construction (`nn.Linear` + PEFT
`modules_to_save`), SimpleAvg dense averaging + classifier stitching — all reimplemented to match
R7's documented behavior, none imported from R7's file.

## 11. Diagnostics

Implemented accumulators + logging functions, shared verbatim by both families (Section 8.7 of the
script): `accuracy_diagnostic_rows` (pre/post-calibration open + restricted, points 1–4, 18),
`kd_teacher_mass_rows` (teacher probability mass on old/current/future *before* masking, point 5),
`kd_loss_rows` (masked KD loss, KD/CE ratio, effective KD weight, points 6–8),
`dense_orth_rows` (raw/weighted DenseOrth, DenseOrth/CE ratio, dense-update norms, effective λ,
points 9–12, 14) plus `pairwise_dense_cosine_matrix()` (point 13), `best_epoch_rows` /
`validation_ce_rows` / `final_accuracy_rows` (points 15–18).

## 12. Success criteria

Encoded as documentation/expectations in the script and this report, not as hard assertions that
would force an outcome (per the request: final accuracy "may be positive, neutral, or negative —
do not force an outcome"). No code in the script enforces a minimum accuracy; success is evaluated
after the (not-yet-run) training completes, against: SimpleAvg-KD-corrected within ~2pp of plain
SimpleAvg with restricted accuracy preserved and antiquity bias reduced; RankExt-KD-corrected
compared against historical RankExt+KD to see whether old-seen masking helps or harms; DenseOrth
(both families) must not dominate CE by orders of magnitude (enforced upstream by the shared-λ
diagnostic's target range, not by training-time clipping) and should not cause severe convergence
delay; combined variants must not inherit SimpleAvg-KD's catastrophic collapse.

## 13. Code audit

`python3 -m py_compile experiments_prepared/supervisor_regularizer_repair_both_families_r8.py` —
**PASS**. `python3 experiments_prepared/supervisor_regularizer_repair_both_families_r8.py --mode
selftest` — **36/36 checks PASS**, covering every item the request asked to verify:

- historical R7 methods/plain methods/old full-logit-KD path/old factor-orth path unchanged — no
  import of the historical script anywhere in this file (checked by scanning this file's own
  source for `import` lines referencing it); R7's own file has zero git diff (Section 14).
- corrected old-seen KD mask exact, current/future classes excluded (all 5 steps, plus a bit-for-bit
  recomputation check against a manual old-seen-only KL computation).
- prior dense deltas detached (`dense_orth_penalty` raises `AssertionError` if a non-detached tensor
  is passed — verified it actually raises).
- current dense delta retains gradient (`current_new_delta()`/live `B@A` both verified to
  backprop).
- RankExt previous blocks remain frozen (`frozen_blocks` entries verified `requires_grad=False`
  immediately after `add_frozen_block_from_current()`).
- no merge autograd path — SimpleAvg's `simple_average_deltas`/`apply_deltas_to_base` operate on
  CPU tensors from `extract_lora_state()` (`.detach().cpu().float().clone()`), structurally
  identical to R7's non-differentiable merge; no loss term reads a merged model's own output.
- warmup applied identically in new methods — verified via source inspection that both trainer
  classes reference the same `KD_WARMUP_EPOCHS/ENABLED` and `ORTH_WARMUP_EPOCHS/ENABLED` constants.
- shared-λ logic explicit — at the time of this audit, `SELECTED_SHARED_LAMBDA is None` was
  verified, and `pick_shared_lambda()` was verified to both (a) find a shared value when the
  synthetic data admits one and (b) report, rather than silently resolve, a genuine conflict.
  **Superseded:** `SELECTED_SHARED_LAMBDA` is now `20.0`, set from the gradient-ratio diagnostic —
  see `thesis_agent/reports/r8_denseorth_formulation_and_gradient_audit.md` Section 8.
- `py_compile` passes (above).

## 13.5. Class-order / KD-mask correctness audit (post-hoc addendum)

Before running the shared-λ diagnostic, `old_seen_class_ids(step_idx) = range(0, step_idx*20)` was
independently re-verified against the actual protocol rather than accepted as a contiguous-range
assumption.

**Traced in the historical script** (`experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`):
`class_splits = [list(range(i*20,(i+1)*20)) for i in range(5)]`, `classes_for_step(step_idx) =
class_splits[step_idx]`; `filter_by_classes()` filters rows by raw `fine_label` id and never
relabels them; `preprocess_train`/`preprocess_val` set `labels = int(fine_label)` unchanged;
`CLIPVisionForCIFAR100.forward()` calls `F.cross_entropy(logits, labels)` against the full
100-way `logits` with these raw labels. **No remapping to per-step-local/contiguous-incremental
indices occurs anywhere** — classifier output index *i* is CIFAR-100 `fine_label` *i*, always, for
every step. `make_train_dataset()`'s own replay-class construction already uses the identical
pattern R8 now uses for `old_seen_class_ids()`:
```
old_classes = []
for old_step in range(step_idx):
    old_classes.extend(classes_for_step(old_step))
```

**Result: `range(0, step_idx*20)` was numerically exact for every step** (verified by
`print_class_id_audit_table()` below), because `classes_for_step()` itself partitions the
already-contiguous native label space into increasing, non-overlapping, unshuffled 20-class
chunks — not because of any remapping step.

| Step | CURRENT | OLD_SEEN (from protocol) | OLD_SEEN (used by KD) | Match |
|---|---|---|---|---|
| 1 | [0..19] | [] | [] | YES |
| 2 | [20..39] | [0..19] | [0..19] | YES |
| 3 | [40..59] | [0..39] | [0..39] | YES |
| 4 | [60..79] | [0..59] | [0..59] | YES |
| 5 | [80..99] | [0..79] | [0..79] | YES |

**R8 was nonetheless refactored** (not left as the coincidentally-correct independent formula):
`classes_for_step()` is now the single source of truth (moved to Section 3, byte-for-byte matching
R7's construction), and `old_seen_class_ids()`/`future_class_ids()` are DERIVED from it via an
explicit union loop (identical in shape to R7's own replay-class loop above), rather than
reimplementing `range()` arithmetic independently. This makes correctness structural — tied
permanently to whatever `classes_for_step()` actually returns — rather than contingent on the
class order staying contiguous by coincidence. Values are unchanged (proven identical above); only
the derivation is now provably, not just numerically, correct. `log_kd_teacher_mass()`'s three
probability-mass buckets (old/current/future) call these same three functions, so they inherited
the fix automatically — confirmed by source inspection in the self-test.

## 14. Historical-method preservation

`git status --short` on `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`
shows **no diff** (file untouched, not even listed as modified). The new file
`experiments_prepared/supervisor_regularizer_repair_both_families_r8.py` is a separate, newly
added, untracked file. `R7/` (the results directory) is untouched (still only untracked, as it was
before this task — never written to by this task). No historical method identifier, config value,
or artifact was read, imported, or overwritten.
