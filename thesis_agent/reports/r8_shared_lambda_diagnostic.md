# R8 Shared-λ Diagnostic Results

**Status: DIAGNOSTIC RUN ONLY. Full R8 training was NOT launched.**

Executed via:
```
python experiments_prepared/supervisor_regularizer_repair_both_families_r8.py \
    --mode lambda_diagnostic --json-out thesis_agent/reports/r8_lambda_diagnostic_results.json
```
Raw run log: `thesis_agent/reports/r8_lambda_diagnostic_run.log`.
Structured numeric results: `thesis_agent/reports/r8_lambda_diagnostic_results.json`.

## 1. What was actually run

The already-implemented `run_lambda_diagnostic()` in
`experiments_prepared/supervisor_regularizer_repair_both_families_r8.py` (Section 8.9), using
real CLIP-ViT-B/16 weights and real CIFAR-100 images (both already cached locally — no new
downloads were needed), with a deliberately **bounded** amount of real training to reach the
"first meaningful DenseOrth-active step" this task's instructions explicitly allow, never
proceeding into the 8-method experiment:

| Family | Phase A: step-1 minimal-state construction | Phase B: step-2 CE-only warm-up (no orth yet) | Phase C: no-update measurement |
|---|---|---|---|
| SimpleAvg | fresh LoRA (r=80), 15 real CE-only optimizer steps, batch=16, capped to a 512-image pool of step-1's classes | fresh LoRA (r=80) for step 2, 10 real CE-only optimizer steps on step-2's classes | 6 forward-only batches, no optimizer step |
| RankExt | fresh rank-16 block, 15 real CE-only optimizer steps on step-1's classes | grown to rank 32 (new 16-rank block), 10 real CE-only optimizer steps on step-2's classes | 6 forward-only batches, no optimizer step |

Phase C is the actual "no-update diagnostic": for each of 6 representative batches, CE and the
raw DenseOrth penalty are computed with **no backward pass and no optimizer step**, using the
exact same `dense_cosine_sq()`/module-averaging code path as
`SimpleAvgCorrectedTrainer`/`RankExtCorrectedTrainer`'s DenseOrth branch. `lambda * raw_orth` and
`weighted_orth / CE` are then computed arithmetically for all 4 candidate λ from that single
measurement (the forward pass does not depend on λ at all, so one measurement suffices for every
candidate — no re-running per λ).

This is explicitly a **reduced-scope** construction relative to R7's full 9-epoch-per-step
protocol (25 total gradient steps per family vs. R7's ~600+ batches/epoch × 9 epochs) — see
Section 4 for why this matters to the result's interpretation.

## 2. Result table

| λ | SimpleAvg DenseOrth/CE | RankExt DenseOrth/CE |
|---|---|---|
| 1 | 1.92e-05 | 1.81e-06 |
| 5 | 9.58e-05 | 9.06e-06 |
| 10 | 1.92e-04 | 1.81e-05 |
| 50 | 9.58e-04 | 9.06e-05 |

Raw measurements (mean over 6 batches):

| Family | CE (mean) | raw DenseOrth (mean) | raw cosine (signed, mean) | current Δ‖·‖ | previous Δ‖·‖ |
|---|---|---|---|---|---|
| SimpleAvg | 3.104 | 5.950e-05 | +0.0070 | 0.0635 | 0.0806 |
| RankExt | 3.445 | 6.241e-06 | −0.0003 | 0.0537 | 0.0602 |

Note: `raw_orth` is **identical across all 6 measurement batches** within each family (this is
expected, not a bug — see Section 5: the DenseOrth penalty is a pure function of the current
model's LoRA/block weights, not of the input batch, so it cannot vary batch-to-batch when no
optimizer step happens between measurements; only CE, which does depend on the batch's actual
images/labels, varies).

## 3. Shared-λ decision

**CONFLICT — no shared λ found among {1, 5, 10, 50}.** Every candidate, for both families, lands
several orders of magnitude *below* the desired [0.1, 1.0] × CE range — the largest candidate
(λ=50) reaches only 9.6e-4 × CE for SimpleAvg and 9.1e-5 × CE for RankExt, roughly 100–1000×
short of the floor of the target range. `pick_shared_lambda()` correctly reported this as a
conflict rather than silently selecting a per-family value (verified: `selected_lambda: null` in
the JSON output).

**CORRECTION (arithmetic error found and fixed by a follow-up audit):** an earlier version of this
report extrapolated the required-λ range from `raw_orth` alone (`0.1/raw_orth`, `1.0/raw_orth`),
which implicitly assumes CE=1 — it never divided by the measured CE. That is wrong: the quantity
that must land in [0.1, 1.0] is `weighted_orth / CE = λ·raw_orth / CE`, so the correct bound is
`λ = ratio_target / (raw_orth / CE) = ratio_target / (ratio measured at λ=1)`. Using the actual
measured `weighted_orth/CE` at λ=1 (Section 2's own table: 1.9167911146532983e-05 for SimpleAvg,
1.8115770848244375e-06 for RankExt — CE was already folded into these; only the *bound
computation* omitted it), the corrected ranges are:

- **SimpleAvg: [5,217, 52,171]**
- **RankExt: [55,201, 552,005]**

**There is NO overlap** between these two corrected ranges (SimpleAvg's upper bound, 52,171, is
still 3,030 below RankExt's lower bound, 55,201) — the previously reported "narrow overlap around
[16,023, 16,807]" was an artifact of the missing CE division and is retracted. This does not change
any measured diagnostic data (CE, raw_orth, raw cosine, norms in Section 2 are exactly as measured
and unchanged) — only the downstream arithmetic that extrapolates from them. See
`thesis_agent/reports/r8_denseorth_formulation_and_gradient_audit.md` for the full follow-up audit,
including why loss-scalar-magnitude matching to CE is itself now in question as the right target
(Section 4 below and that report's Sections 2–4).

## 4. Important limitation: this measurement is from an early-training regime

The measured `raw_orth` values (5.95e-05 for SimpleAvg, 6.24e-06 for RankExt) are **substantially
smaller** than what R7's own fully-trained (9-epoch) SimpleAvg deltas actually exhibit: the R7
forensic report's merge-mechanism data (`thesis_agent/reports/r7_simpleavg_kd_factororth_forensic_analysis.md`,
Section 10) measured a mean `cos(dW_1, dW_t)` of ≈0.02–0.04 across fully-trained steps, i.e. a raw
cosine² of ≈0.0004–0.0016 — roughly **7–27× larger** than the 5.95e-05 measured here after only 25
gradient steps. This is expected: with only 15+10 optimizer steps (vs. R7's ~600+ batches/epoch ×
9 epochs), the LoRA/rank-block weights have moved only a small distance from their
near-zero-delta initialization (PEFT LoRA and `GrowingRankLoRALinearDenseOrth` both zero-init the
`B` factor), so both the deltas' magnitudes and their overlap with the previous step's delta are
still far from their eventual, fully-trained values.

**Consequence for the shared-λ question:** the true required λ once training is representative
is very likely *smaller* than the (corrected) ~5,200–552,000 range above (since `raw_orth` should
grow as training progresses further), but there is no principled way to extrapolate exactly how
much smaller from this bounded measurement alone — and even a 7–27× correction would still land
the required λ in the hundreds-to-low-thousands, still far outside {1, 5, 10, 50}. **The conflict
finding is robust to this limitation** (no candidate in the tested set is remotely close for
either family), but the specific "required λ" numbers in Section 3 should be read as
order-of-magnitude context from an admittedly under-trained probe, not as a calibrated target.

**Deeper issue, raised by the follow-up audit (see the dedicated report):** even setting the
arithmetic error aside, matching DenseOrth's *scalar loss value* to 0.1–1.0×CE is itself now in
question as the right calibration target — the measured raw (signed, unsquared) cosine is tiny
(+0.0070 for SimpleAvg, −0.0003 for RankExt), consistent with two independently-trained,
high-dimensional dense updates being close to incidentally orthogonal already, not with a
substantial redundancy that a loss term needs to fight. Forcing a near-zero cos² to become
0.1–1.0×CE via λ in the tens-to-hundreds-of-thousands would manufacture a large loss value out of
what may be mostly geometric noise. `thesis_agent/reports/r8_denseorth_formulation_and_gradient_audit.md`
replaces loss-magnitude matching with a **gradient-norm** diagnostic (comparing `‖g_orth‖` to
`‖g_ce‖` directly, on the same trainable parameters) as the more defensible signal, and audits
whether the normalized-cosine-squared formulation itself is the right one to carry forward.

## 5. Normalization check

Verified directly (`check_dense_orth_scale_invariance()`, using the actual measured norms):
rescaling the current delta by 3.7× leaves `cos²` unchanged for **both** families
(`scale_invariance_ok: {"simple_avg": true, "rank_extension": true}`). This confirms the
DenseOrth formula is genuinely scale-invariant as designed — `cos = <a,b> / (‖a‖‖b‖ + eps)`
multiplies both numerator and denominator by the same factor when `a` is rescaled, so the ratio
(and its square) is unchanged. **Lambda selection is not being driven by an unnormalized raw
delta-magnitude bug** — the tiny measured ratios in Section 2 reflect a genuinely small
cosine-squared overlap at this training stage (consistent with two independently-initialized,
lightly-trained, high-dimensional weight updates being close to incidentally orthogonal), not a
broken normalization.

## 6. Decision

Per the explicit instruction ("If no shared lambda is scientifically reasonable: STOP and report
that explicitly. Do not start R8 training."): **stopping here.**
`SELECTED_SHARED_LAMBDA` remains `None` in
`experiments_prepared/supervisor_regularizer_repair_both_families_r8.py` — not set to any
candidate, not set to a family-specific value, and not set to an extrapolated value outside the
tested set. `--mode train` therefore continues to refuse to run (verified, Section 7). Full R8
training was not launched.

**Superseded by:** `thesis_agent/reports/r8_denseorth_formulation_and_gradient_audit.md`, which
(a) re-runs the diagnostic at a more representative training state, (b) replaces the loss-ratio
target with a gradient-norm-ratio target, and (c) audits whether the normalized cosine-squared
DenseOrth formula is even the right one to carry forward, rather than continuing to test only
{1, 5, 10, 50} against a loss-magnitude target.

## 7. Verification

```
$ python -m py_compile experiments_prepared/supervisor_regularizer_repair_both_families_r8.py
PYCOMPILE_OK

$ python experiments_prepared/supervisor_regularizer_repair_both_families_r8.py --mode selftest
R8 CODE-SAFETY SELF-TEST (43/43 passed)
OVERALL: PASS

$ python experiments_prepared/supervisor_regularizer_repair_both_families_r8.py --mode train
Refusing to start training: no shared lambda has been selected. ...
```

`git status --short` on `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py` and
on `thesis_writing/chapter_1/`, `thesis_writing/chapter_2/`, `thesis_writing/chapter_3/` shows no
changes from this task.
