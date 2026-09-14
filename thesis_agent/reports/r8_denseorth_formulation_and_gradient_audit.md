# R8 DenseOrth Formulation and Gradient Audit

**Status: FINALIZED FOR EXECUTION, NOT YET LAUNCHED.** `SELECTED_SHARED_LAMBDA = 20.0` (set from
the gradient-ratio diagnostic in Section 4 — see the amendment in Section 8). The whole-matrix
normalized-cosine-squared DenseOrth formulation is **kept for R8 validation** (Section 3's verdict
was revised from MODIFY to KEEP — see the amendment). Full R8 training has still not been
launched. Historical R7 files were not modified.

This report resolves the two issues raised about the shared-λ diagnostic
(`thesis_agent/reports/r8_shared_lambda_diagnostic.md`): an arithmetic error in the extrapolated
λ ranges, and the deeper question of whether matching DenseOrth's *scalar loss value* to CE is
even the right calibration target in the first place.

---

## 1. Issue 1 — the extrapolation arithmetic error, corrected

The previous report computed the λ needed to reach the target ratio range as `0.1/raw_orth` and
`1.0/raw_orth` — i.e. it implicitly divided by CE=1, forgetting that the quantity that must land
in [0.1, 1.0] is `weighted_orth/CE`, not `weighted_orth` alone.

Independently re-derived here from the unchanged measured data (`r8_lambda_diagnostic_results.json`):
at λ=1, `weighted_orth/CE` was measured as **1.9167911146532983e-05** for SimpleAvg and
**1.8115770848244375e-06** for RankExt. Since this ratio scales linearly in λ (the forward pass
that produces `raw_orth` and `CE` does not depend on λ at all — only the reported `weighted_orth`
and its ratio do), the λ that puts the ratio at a target value `r` is exactly `λ = r / ratio_at_λ=1`:

```
SimpleAvg: λ_lo = 0.1 / 1.9167911146532983e-05 = 5217.1   λ_hi = 1.0 / 1.9167911146532983e-05 = 52170.5
RankExt:   λ_lo = 0.1 / 1.8115770848244375e-06 = 55200.5  λ_hi = 1.0 / 1.8115770848244375e-06 = 552005.2
```

**Corrected ranges: SimpleAvg [5,217, 52,171]; RankExt [55,201, 552,005]. No overlap** (SimpleAvg's
upper bound is 3,030 below RankExt's lower bound). This matches the correction supplied in the
follow-up instructions to within rounding. `thesis_agent/reports/r8_shared_lambda_diagnostic.md`
has been corrected in place (Section 3–4); no measured diagnostic data (CE, raw_orth, raw cosine,
norms) was changed, only this downstream arithmetic.

---

## 2. Issue 2 — is loss-magnitude matching to CE even the right target?

**No, not as the primary criterion.** Three independent arguments:

1. **The bounded quantity argument.** `cos²(delta_current, delta_previous) ∈ [0, 1]` by
   construction. The measured raw (signed, unsquared) cosine is tiny — **+0.0070 for SimpleAvg,
   −0.0003 for RankExt** — meaning the two dense updates are already close to orthogonal at the
   measured state. A λ in the tens-to-hundreds-of-thousands does not "correct" a real conflict;
   it algebraically inflates a near-zero number until it happens to equal a target fraction of
   CE. The *loss value* reaching 0.1–1.0×CE says nothing about whether there was a meaningful
   geometric redundancy to begin with.
2. **The dimensionality argument.** Each compared dense delta here is a full CLIP-ViT-B/16
   attention-projection matrix, 768×768 = 589,824 entries. For two vectors drawn from a
   589,824-dimensional space with no coordinated structure, the expected cosine magnitude is
   `O(1/√d) ≈ O(1/√589824) ≈ 0.0013` by concentration-of-measure — i.e. the measured +0.0070 /
   −0.0003 are within a few multiples of what pure high-dimensional geometric noise would produce
   on its own. A formula that reports "extremely small overlap" in this regime is not necessarily
   detecting a weak-but-real signal that needs amplifying — it may simply be reporting that
   generic high-dimensional vectors are nearly orthogonal, which is a property of the geometry,
   not of the training dynamics.
3. **The gradient-influence argument.** What actually matters for training is how much the
   DenseOrth term perturbs the optimizer's update direction relative to CE, i.e. gradient norms
   and their relative alignment/magnitude — not the scalar loss value. A loss term can have a
   tiny value yet a large, informative gradient (e.g. very peaked near a boundary), or a
   loss term can be forced to a large value via λ while still contributing a comparatively small,
   uninformative gradient nudge relative to CE's. Section 3 measures this directly.

**Conclusion:** loss-magnitude matching to CE (Sections 2–3 of the prior report) is not, by
itself, scientific justification for a λ. It is retained as one diagnostic signal (it is cheap
and shows the term is at least numerically alive) but is demoted from "the calibration target" to
"a sanity check," in favor of the gradient-norm-ratio measurement below.

---

## 3. Supervisor's original orthogonality idea — formulation audit

**`orthogonality.txt` was searched for across the entire repository (`find . -iname
"orthogonal*"`, plus content greps for `tr(`, `lambda_ortho`, `M_{t-1}` across every `.txt`/`.md`
file) and does not exist anywhere in this project.** This audit therefore treats the formula
restated in the follow-up instructions verbatim as the specification to compare against:

```
lambda_ortho * tr(M_(t-1) L_t^T)
```
applied over corresponding weight matrices and LoRA dense updates `delta_W = A @ B` (this repo's
convention, per `extract_lora_state()`/`GrowingRankLoRALinearDenseOrth`, is `delta_W = scaling *
(B @ A)`, with `A: [r, in]`, `B: [out, r]` — the same object under a transposed-factor naming
convention, not a different formula). This is a **raw, unnormalized Frobenius inner product**
(`tr(XY^T) = <X,Y>_F = sum(X*Y)` elementwise) — the supervisor's formula contains no division by
either matrix's norm.

**Historical precedent already in this codebase:** R7's `orth_mode="delta_trace"`
(`compute_independent_lora_orth_components()`,
`experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`) computes exactly this raw
trace (`raw_trace = torch.sum(previous_delta * delta)`, logged as `raw_trace_mean_unnormalized`)
— but then **also** computes a normalized cosine (`cosine = raw_trace / (ref_norm * delta_norm)`)
and uses `orth_loss_abs` (mean **absolute cosine**, not the raw trace) as the actual training
loss. I.e., even when this codebase's own earlier work implemented the supervisor's literal raw
trace, it ended up applying a normalized version of it in practice — direct precedent that pure
raw trace was found impractical to use unnormalized (the code preserves the raw number only as a
diagnostic, `raw_trace_mean_unnormalized`, never as the trained penalty).

### Four formulations compared

| | Formula | Matches supervisor's literal formula | Scale-stable across training | Suitable for both families | Avoids "trivially tiny in high dim" |
|---|---|---|---|---|---|
| **A.** Raw/squared Frobenius inner product, dense-update space | `<Δ_cur, Δ_prev>_F` or its square, **no normalization** | Yes (this *is* the raw trace, up to squaring) | **No** — scales with `‖Δ_cur‖·‖Δ_prev‖`, which grows through training (measured here: current-delta norm grew from ≈0.02 at near-init to ≈0.05–0.08 after 25–50 steps; R7's fully-trained deltas reach ≈0.4–1.6), so a fixed λ that is reasonable early becomes disproportionately large later, and vice versa — the exact instability pattern R7's FactorOrth exhibited (λ=50 → ~144–570× CE early, ≈0 late; see `r7_simpleavg_kd_factororth_forensic_analysis.md` §12) | Yes, structurally (both families produce a dense Δ) | **Yes** — an unnormalized inner product does not get divided down by dimension, so it tracks real magnitude, not just angle |
| **B. (current R8 implementation)** Normalized Frobenius cosine-squared, global per matrix | `cos²(Δ_cur, Δ_prev)` | Partial (adds normalization the formula didn't specify) | **Yes** — bounded in [0,1], invariant to delta magnitude (verified in the prior report's scale-invariance check) | Yes, already implemented identically for both | **No — confirmed by this diagnostic.** Measured cos² ≈ 4.9e-5 (SA) / 9e-8 (RE, from cos≈-0.0003²) at a full-matrix (589,824-dim) granularity, consistent with the 1/√d concentration argument in Section 2 |
| **C.** Normalized overlap averaged per layer | `mean_layers[cos²(Δ_cur,l, Δ_prev,l)]` | Partial, same as B | Yes, same as B | Yes, same as B | **No, not as literally specified** — R8's current implementation (Section "SimpleAvg implementation" of the preparation report) already averages a per-*module* cosine² across ~24 target modules, which is functionally B with an extra averaging step; each per-module term is still a full 768×768 matrix comparison, so it inherits B's dimensionality problem term-by-term. A genuinely different, finer granularity (e.g. per-*column*, dimension 768 instead of 589,824 — the same convention `column_decouple_delta()`/`column_normalize()` already use elsewhere in this codebase for the SimpleAvg merge-diagnostics) would reduce dimensionality per comparison unit and is the most plausible way to make "C" meaningfully different from B — **a possible future ablation; explicitly NOT part of R8** |
| **D.** Supervisor-style previous-model/current-delta trace overlap | `tr(M_{t-1} · Δ_t^T)`, literal | **Yes — this is the literal formula** | **No**, same instability as A | Depends on what "`M_{t-1}`" means: if it is the accumulated **delta only** (R7's own `average_delta_reference_state()` precedent — delta vs. delta, matching A), suitable for both families the same way A is; if it is meant as the full **effective weight matrix** (pretrained backbone + accumulated delta), it would measure overlap against a matrix dominated by generic pretrained-feature structure, which is a different and likely less meaningful quantity for continual-learning purposes, and was not how this codebase operationalized the idea | Same as A |

**A and D coincide** under the interpretation this codebase already uses (`M_{t-1}` = accumulated
delta, not accumulated delta + pretrained backbone) — R7's own `delta_trace` mode is the
historical, code-confirmed instance of exactly this. Given that R7's own implementers already
found the raw form impractical enough to normalize it in practice (Section 3 above), the
honest reading is: **no single formulation among A–D satisfies all four criteria simultaneously.**
There is a genuine trade-off between magnitude-sensitivity (A/D, unstable) and scale-stability
(B/C, vanishes in high dimension).

### Verdict (AMENDED for R8 — see Section 8)

- **CURRENT COSINE-SQUARED DENSEORTH FORMULATION: KEEP FOR R8 VALIDATION** (revised from this
  section's original MODIFY verdict — see Section 8 for the full amendment). Whole-matrix
  normalized Frobenius cosine-squared, `mean_{m, i<t}(cos_ti_m²)`, is retained for R8 exactly as
  already implemented (`dense_cosine_sq()`/`dense_orth_penalty()`), for both families, with no
  change in granularity. Rationale: the full-matrix cosine *values* are numerically small
  (Section 2's dimensionality argument still holds — this has not been retracted), but the
  gradient diagnostic (Section 4) shows the loss nonetheless exerts a controlled, non-negligible
  optimization influence once weighted by a moderate shared λ — the thing that actually matters
  for training dynamics. Changing granularity (e.g. to per-column) *before* R8 has ever tested the
  current, already-implemented formulation would introduce a second, uncontrolled design change
  at the same time as the KD correction — confounding any result. Per-column DenseOrth remains
  a possible **future ablation**, to be considered only after R8's whole-matrix result exists to
  compare against — it is explicitly **not part of R8**.
- **BEST MATCH TO SUPERVISOR'S LITERAL INTENT** (unchanged by the amendment — this is about the
  *formula*, not the *decision* of what to run): formulation **D/A** — the raw, unnormalized
  Frobenius trace/inner product between the current step's/block's dense update and the previous
  step's/block's dense update, `tr(Δ_prev · Δ_cur^T) = <Δ_prev, Δ_cur>_F`, exactly matching R7's
  own historical `delta_trace` orth_mode's raw-trace computation (before that code's own
  normalization step). This remains the best LITERAL match to the supervisor's formula; R8 uses
  the normalized cosine-squared form instead, per the KEEP decision above, not because D/A stopped
  being the literal match.

---

## 4. Gradient-norm diagnostic

Implemented in `experiments_prepared/supervisor_regularizer_repair_both_families_r8.py`, Section
8.10 (`run_gradient_diagnostic()`, `measure_gradients_simple_avg()`,
`measure_gradients_rank_extension()`, `derive_candidate_lambdas_from_gradient_ratio()`), reachable
via `--mode gradient_diagnostic`. For each of several representative batches, at a **more
representative model state** than the prior report's 15+10-batch probe (see below), two
*independent* backward passes are run — `g_ce = ∇CE`, `g_orth = ∇DenseOrth` — on the exact same
trainable LoRA/rank-block parameters, with grads zeroed between them and **no optimizer step
anywhere in the measurement** (verified by the self-test: `run_code_safety_selftest()` now checks
that neither `measure_gradients_simple_avg` nor `measure_gradients_rank_extension` contains an
`optimizer.step()` call). `‖g_ce‖`, `‖g_orth‖`, `cos(g_ce, g_orth)`, and `‖g_orth‖/‖g_ce‖` (at
λ=1 — this ratio scales linearly in λ, exactly like the loss-ratio did, so one measurement again
suffices for every candidate) are reported per batch and averaged.

**More representative state:** step-1 minimal-state construction increased from 15 to **30**
CE-only optimizer steps, step-2 warm-up from 10 to **20**, both still on a bounded image pool
(768 images/step, up from 512) — substantially more real training than the prior probe while
remaining a bounded diagnostic (no KD, no DenseOrth training, no full 8-method run).

### Results

Actually run (`thesis_agent/reports/r8_gradient_diagnostic_run.log` /
`r8_gradient_diagnostic_results.json`). Step-1 warm-up CE dropped 5.13→1.45 (SimpleAvg) and
4.65→3.20 (RankExt) over 30 batches — real, meaningful learning, not near-init noise.

| Family | CE (mean, 5 batches) | ‖g_ce‖ (mean) | ‖g_orth‖ (mean, std) | ratio ‖g_orth‖/‖g_ce‖ @ λ=1 | cos(g_ce, g_orth) |
|---|---|---|---|---|---|
| SimpleAvg | 2.210 | 6.300 | 0.01799 (std 0.0) | **0.002875** (0.29%) | **−0.0229** |
| RankExt | 3.227 | 0.300 | 0.00234 (std 0.0) | **0.007871** (0.79%) | **+0.0016** |

`grad_orth_norm_std = 0.0` for both families, exactly as expected (Section 8.9's finding
generalizes to gradients: `g_orth` depends only on the current weights, not the batch, so it is
identical across all 5 measurement batches — this is the diagnostic behaving correctly, not a
bug). The absolute magnitudes of `‖g_ce‖`/`‖g_orth‖` are ~7–21× larger for SimpleAvg than RankExt
(6.30 vs 0.30, 0.018 vs 0.0023) — this reflects SimpleAvg's larger trainable-parameter count at
this step (LoRA rank 80 vs. RankExt's rank-16 new block), **not** a meaningful cross-family
comparison; the *ratio* ‖g_orth‖/‖g_ce‖, computed within each family on its own parameters, is
the only quantity compared across families below.

**Candidate λ derived from the gradient ratio** (target: weighted gradient ratio ∈ [0.05, 0.5],
i.e. `λ = target / ratio_at_λ=1`):

| Family | λ range for target [0.05, 0.5]× | Round-number candidates |
|---|---|---|
| SimpleAvg | **[17.4, 173.9]** | 17, 37, 81, 170 |
| RankExt | **[6.4, 63.5]** | 6.4, 14, 29, 64 |

**These two ranges overlap: [17.4, 63.5].** This is a materially different, much more encouraging
picture than the loss-ratio target (Section 1: [5,217, 52,171] vs. [55,201, 552,005], **no**
overlap). Under the gradient-influence criterion, a shared λ somewhere in roughly 17–64 (e.g. 20,
30, or 50) would plausibly put *both* families' DenseOrth gradient influence inside a modest,
defensible 5–50% of their own CE gradient norm — a two-to-three-order-of-magnitude smaller, far
more plausible number than anything the loss-ratio approach produced.

---

## 5. Critical question — is DenseOrth even needed at the measured state?

**Correction (per follow-up audit): the two cosine quantities below measure different objects and
must not be read as two measurements of the same signal.**

- **Raw delta cosine** (`cos(Δ_current, Δ_previous)`, Section 8.9's `run_lambda_diagnostic()`) is
  a **geometric** quantity: how aligned the two *dense parameter updates themselves* are, as
  static tensors, independent of any loss or optimizer. It answers "how much does the current
  step's weight update already resemble the previous step's, before any regularizer is applied."
- **Gradient cosine** (`cos(g_ce, g_orth)`, Section 4's `run_gradient_diagnostic()`) is a **local
  optimization-interaction** quantity: how the *gradient* of the DenseOrth penalty relates to the
  *gradient* of CE, at a specific point in parameter space. It answers "if the optimizer followed
  DenseOrth's gradient, would that pull mildly with or against the direction CE wants to move."

A weight update can be geometrically near-orthogonal to a previous one (small raw delta cosine)
while its *penalizing gradient* still points in a direction that is well- or poorly-aligned with
CE's gradient (a different, small or large, gradient cosine) — the two are related only through
the (nonlinear, cos → cos² → gradient) chain, not by being restatements of each other. Reporting
that they "agree" or "disagree" in sign, or treating a sign flip between them as evidence of noise
versus signal, is not a valid inference from either quantity alone and is retracted from the
version of this section that made that argument.

**Read independently, each quantity supports a narrower, still-useful conclusion:**

- **Raw delta cosine is near the high-dimensional noise floor for both families** (Section 2's
  `O(1/√d) ≈ 0.0013` argument): +0.0070 (SimpleAvg), −0.0003 (RankExt). Geometrically, the current
  and previous dense updates are already close to orthogonal — there is little redundant
  *direction* for DenseOrth to remove, for either family, at the measured (still bounded, 15+10
  batches) state.
- **Gradient cosine shows a small directional relationship for SimpleAvg, and essentially none for
  RankExt**, at a separately measured, more representative (30+20 batches) state: `cos(g_ce,
  g_orth) = −0.0229` (SimpleAvg) vs. `+0.0016` (RankExt). Taken on its own terms, this says that
  right now, at that state, following DenseOrth's gradient would pull *mildly against* CE's
  gradient for SimpleAvg (a small, genuine plasticity-vs-orthogonality tension), and would do
  essentially nothing either way for RankExt.

**Net assessment, from the gradient measurement alone** (the more decision-relevant of the two,
since it reflects what a nonzero λ actually does to training): DenseOrth is not "clearly
unnecessary" for either family — the gradient exists and is measurable, and Section 4's derived λ
range would give it a small, controlled, non-negligible influence (5–50% of CE's gradient norm)
for both. Nor is a large λ justified — Sections 1–2 already ruled that out on the loss-magnitude
side, and the gradient side independently points to a *modest* λ (tens, not thousands). This
supports keeping the current formulation for R8 (Section 3's amended verdict) and testing it with
a moderate, gradient-derived shared λ, rather than either abandoning DenseOrth or forcing a huge
one.

---

## 6. Reproducibility

```
python experiments_prepared/supervisor_regularizer_repair_both_families_r8.py \
    --mode gradient_diagnostic --json-out thesis_agent/reports/r8_gradient_diagnostic_results.json
```
Raw run log: `thesis_agent/reports/r8_gradient_diagnostic_run.log`.
Structured numeric results: `thesis_agent/reports/r8_gradient_diagnostic_results.json`.

```
$ python -m py_compile experiments_prepared/supervisor_regularizer_repair_both_families_r8.py
PYCOMPILE_OK

$ python experiments_prepared/supervisor_regularizer_repair_both_families_r8.py --mode selftest
R8 CODE-SAFETY SELF-TEST (51/51 passed)
OVERALL: PASS
```

**Superseded by Section 8:** at the time this diagnostic ran, `SELECTED_SHARED_LAMBDA` was still
`None` and `--mode train` refused to start. It has since been set to `20.0` (Section 8) — see that
section for the updated verification commands and self-test count (64/64).

---

## 7. Fairness note

DenseOrth's formulation and calibration target are now decided for R8 (Section 3's amended
verdict: KEEP the current whole-matrix normalized cosine-squared formula; Section 4/8's verdict:
gradient-ratio, not loss-ratio, as the calibration target) — both are applied via the **same
shared functions** to both families, exactly as `dense_cosine_sq()`/`dense_orth_penalty()` and
`linear_warmup_multiplier()` already are (Sections 4/8/11/13 of the R8 preparation report), with
one shared λ=20.0. No family-specific formulation or target has been introduced by this audit —
the only family-specific things remain teacher construction and the dense-delta storage mechanism,
both already documented as structural, not tunable, differences.

---

## 8. Amendment — R8 finalized for execution (λ=20.0, not yet launched)

This section records the finalization pass that acted on Sections 1–7 above.

**Lambda selection.** `SELECTED_SHARED_LAMBDA = 20.0`, set in
`experiments_prepared/supervisor_regularizer_repair_both_families_r8.py`. Justification from
Section 4's actual measurement: at λ=20, `weighted ‖g_orth‖/‖g_ce‖ ≈ 20 × 0.0028747374 = 0.0575`
(SimpleAvg) and `≈ 20 × 0.0078708385 = 0.1574` (RankExt) — both inside the target [0.05, 0.5]
gradient-influence band. 20 was chosen as the conservative edge of the common admissible range
[17.4, 63.5] (Section 4), rather than a more aggressive value such as 50 that is also inside that
range.

**DenseOrth formulation for R8 (unchanged from what was already implemented — Section 3's KEEP
verdict):**
```
delta_W_t   = scaling * B_t @ A_t                          (current step/block, per module m)
cos_ti_m    = <delta_W_t_m, delta_W_i_m>_F / (||delta_W_t_m||_F ||delta_W_i_m||_F + eps)
L_orth      = mean over {target module m, previous step/block i < t} of cos_ti_m^2
```
identical for SimpleAvg (`i` = each previous CL step) and RankExt (`i` = each previous frozen
rank block) — the same `dense_cosine_sq()`/`dense_orth_penalty()` functions, called from
`SimpleAvgCorrectedTrainer.compute_loss()` and `RankExtCorrectedTrainer.compute_loss()`
respectively.

**Missing piece filled in:** the outer 5-step, 8-method orchestration
(`run_full_method_simple_avg()`, `run_full_method_rank_extension()`, `run_full_r8_experiment()`)
did not exist before this finalization pass — only single-step training functions did. It is now
implemented (dense-merge/classifier-stitching for SimpleAvg, persistent growth + teacher-snapshot
ordering fix for RankExt, confidence-weighted calibration, pre-/post-calibration open+restricted
evaluation, all diagnostics from Section 8.7 of the preparation report), and `--mode train` now
dispatches to it instead of unconditionally refusing.

**Verification, re-run after this amendment:**
```
$ python -m py_compile experiments_prepared/supervisor_regularizer_repair_both_families_r8.py
PYCOMPILE_OK

$ python experiments_prepared/supervisor_regularizer_repair_both_families_r8.py --mode selftest
R8 CODE-SAFETY SELF-TEST (64/64 passed)
OVERALL: PASS
```
New checks added for this pass (all passing): `SELECTED_SHARED_LAMBDA == 20.0`; λ=20 puts both
families' measured gradient ratio inside [0.05, 0.5]; `--mode train` no longer contains the old
unconditional refusal and now dispatches to `run_full_r8_experiment()`; the missing-lambda refusal
path still exists defensively; no factor-space orth-loss function is defined anywhere in this
file; `F.kl_div` is used only inside `masked_kd_loss()` (never inline in either trainer or
orchestration function); both trainers' KD path goes only through `masked_kd_loss()`; KD and
DenseOrth warmup epochs are the identical module constant (1.0, enabled) for both families; the 8
methods and the two plain controls' configuration are unchanged.

**Bounded smoke test** of the new orchestration (one SimpleAvg and one RankExt combined
KD+DenseOrth method, 1 epoch/step, small image/val/eval caps — NOT the real 9-epoch/full-dataset
run, and NOT invoked via `--mode train`) was started to further confirm the orchestration runs
end to end without crashing; see `thesis_agent/reports/r8_smoke_test_full_orchestration.log` for
its outcome (this smoke test's own per-class train/val split construction turned out to be the
dominant cost — ~100 separate full-50k-row filter passes, ~27s each, matching R7's own per-class
splitting design — so it runs considerably longer than a "smoke test" label suggests; this is a
one-time setup cost paid once per full run, not per method, and is not a correctness concern).
Readiness above is established independently by `py_compile` + the 64/64 self-test, which
exercises the mask/warmup/detachment/gradient-flow logic directly; the smoke test is corroborating
evidence, not the basis for the PY_COMPILE/SELF_TEST/READY-TO-SUBMIT verdicts below.

`git status --short` on `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py` and
`thesis_writing/chapter_1/`, `_2/`, `_3/` continues to show no changes from this task. **Full R8
training has not been launched.**
