# Final 5×20 Performance Recovery Plan

**Scope: design and implementation-preparation only.** No training was run, no
evaluation was run, no SLURM job was submitted, no smoke test was executed.
This document synthesizes existing repository evidence (four prior reports,
two source scripts, and the result CSVs they produced) into the design
rationale for `experiments_prepared/final_9method_5x20_performance_recovery.py`
and its launcher, `experiments_prepared/slurm/final_9method_5x20_performance_recovery.sbatch`.

---

## 1. Objective

Recover — and, if the evidence genuinely supports it, exceed — the historical
Exp1/R7 RankExt ceiling of **70.72% all_seen** (FactorOrth λ50 + full-100-way
KD, 9 epochs), while keeping SimpleAvg's already-strong architecture
essentially unchanged and testing the one remaining rational SimpleAvg
lever (a KD temperature ablation). This plan operationalizes the
recommendations of `R7/final_recovery_and_improvement_design_audit.md` into
a concrete, exactly-specified 9-method training script.

---

## 2. Protocol Lock

CIFAR-100, 5 continual-learning steps, 20 classes/step, 100 total classes,
contiguous native-label-order class splits (step1=0–19, step2=20–39,
step3=40–59, step4=60–79, step5=80–99), RankExt cumulative rank schedule
`[16,32,48,64,80]`. No other protocol (20×5, 4×25, wide-rank capacity) is in
scope. These are enforced as hard, unconditional assertions in the production
script — see `R7/final_9method_5x20_static_readiness.md` Section 3 for the
verification that they hold.

---

## 3. Historical 5×20 Results

From `R7/final_recovery_and_improvement_design_audit.md` (Exp1/job 4961962 —
the modern-codebase replication that reproduces the original Exp1 numbers
bit-for-bit):

| Family | Method | all_seen (%) |
|---|---|---:|
| SimpleAvg | plain | 75.16 |
| SimpleAvg | +full KD | 62.98 |
| SimpleAvg | +FactorOrth | 74.29 |
| SimpleAvg | +FactorOrth+full KD | 69.40 |
| RankExt | plain | 35.31 |
| RankExt | +full KD | 64.44 |
| RankExt | +FactorOrth50 | 42.21 |
| **RankExt** | **+FactorOrth50+full KD** | **70.72** |

9 epochs/CL step, full-100-way KD (T=2, weight=1.0, no warmup), FactorOrth
λ=50 (1-epoch warmup), `protect_weight=30.0` active on both KD-bearing
RankExt methods, new-block output warmup disabled on those same two methods,
and a broken (`.copy_()` advanced-indexing no-op) classifier-row restoration.

---

## 4. Current 5×20 Results

From `R7/final_8arm_job4970580_forensic_analysis.md` (job 4970580, 6 epochs,
corrected classifier restoration, no `protect_weight`, old-seen-only KD):

| Family | Method | all_seen (%) |
|---|---|---:|
| SimpleAvg | plain | 75.98 |
| SimpleAvg | KD1, no warmup (job 4969059) | 75.05 |
| SimpleAvg | KD1, 1-epoch warmup (job 4969059) | 76.66 |
| SimpleAvg | KD0.5, no warmup (job 4970580) | 75.77 |
| SimpleAvg | DenseOrth20 | 75.11 |
| SimpleAvg | DenseOrth20+KD1 warmup (job 4969059) | ≈76.19 |
| RankExt | plain | 30.56 |
| RankExt | old-seen KD1.5, 1-epoch warmup | 48.53 |
| RankExt | FactorOrth50 | 40.37 |
| **RankExt** | **FactorOrth50+old-seen KD1.5** | **53.13** |

The two unchanged controls (`simple_avg`, `simple_avg_dense_orth_lam20`)
reproduce job 4969059's numbers **bit-for-bit** (75.98=75.98, 75.11=75.11),
confirming the pipeline is deterministic and that every delta above the plain
arms is attributable to the treatment, not run-to-run noise.

---

## 5. RankExt Historical Recipe

Read directly from `R7/vit_lora_cifar100_full5step_n5.py` and confirmed
against `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`
(the script that reproduces 70.72% exactly, job 4961962):

- **9 epochs/CL step.**
- **Full-100-way KD**: `F.kl_div(F.log_softmax(student_logits/T), F.softmax(teacher_logits/T), reduction="batchmean") * T**2`, no masking to old-seen classes. T=2, weight=1.0, **no KD-weight warmup** (the mechanism didn't exist in Exp1's era).
- **FactorOrth λ=50**: normalized A/B factor cosine-overlap between the persistent adapter's accumulated frozen blocks and the current new block, 1-epoch λ warmup.
- **`protect_weight=30.0`** ("OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION"): an orthonormal basis `P_old` of the frozen teacher's old-class classifier-row span (SVD of mean-removed, row-normalized rows), penalizing the squared projection of the student-minus-teacher CLS-feature drift onto that subspace. Active **only** on the two KD-bearing RankExt methods.
- **New-block output warmup disabled** for those same two KD-bearing methods (`RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS`) — the newly-appended rank block contributes at full strength from batch 1, even while KD is also active.
- **Broken classifier-row restoration** (`tensor[idx].copy_(value)` — a silent no-op under PyTorch's advanced-indexing semantics). **Not reproduced** in the new script — see Section 15.

---

## 6. RankExt Current Recipe

`experiments_prepared/final_8arm_family_specific_5x20.py`'s
`DeltaOrthRankExtensionTrainer` was found, on direct inspection, to **already
contain a byte-for-byte copy** of every mechanism in Section 5 above except
the epoch count and the classifier-restoration bug — carried over verbatim by
an earlier port and simply never activated for job 4970580's 8 arms (whose
method names were deliberately kept outside
`RANKEXT_PROJECTED_PROTECT_METHODS`/`RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS`).
Job 4970580 itself used: 6 epochs, **old-seen-only KD** (`kd_class_scope="old_seen"`,
weight=1.5, 1-epoch warmup), FactorOrth λ=50 (unchanged), `protect_weight=0`
(inactive), new-block warmup **enabled** for all 4 RankExt arms, and the
**fixed** (`tensor[idx] = value`) classifier restoration.

**This is the single most important discovery shaping this script's
construction**: recovering the historical mechanism required no new trainer
code — only configuring two new method entries to activate what was already
there. See `R7/final_9method_5x20_static_readiness.md` Section 9–11 for the
line-level confirmation.

---

## 7. SimpleAvg Current Recipe

Unchanged from job 4970580/4969059: independent, freshly-initialized
per-step LoRA specialists (rank=80, alpha=160, scaling=2, q_proj/v_proj),
arithmetic dense-delta merge (`simple_average_deltas`), classifier row
stitching per step, `confidence_weighted_regime_grouped` calibration, no
replay, no DO-Merge. This script changes only: epoch count (6→9), KD weight
for the KD arms (0.5→1.0, restoring job 4969059's original, better-performing
value), and adds one new arm (T=4 KD ablation).

---

## 8. Full KD Evidence

Full-100-way KD constrains the **relative scale between old-class and
new-class logits** (it matches the student's entire 100-way softmax to the
teacher's), while old-seen KD only matches probability mass *within* the
already-seen classes and contains zero information about how new-class
logits should sit relative to old ones. This is a direct mechanistic match to
the failure mode the prior forensic analysis diagnosed: RankExt's old-task
*restricted* accuracy is already excellent (88–95%) under old-seen KD, but
*open* accuracy for the oldest tasks collapses to single digits — a
classifier/logit recency-bias problem, not a representation-forgetting
problem. Full KD is the only mechanism in this design that directly
addresses that specific axis. Independent, cross-protocol corroboration: a
structurally different CL protocol (R6, 4×25 steps, 7 epochs, also using
full-100-way KD) independently lands RankExt+FactorOrth+KD at 72–73%.

## 9. Old-Seen KD Evidence

Old-seen KD (job 4970580) is confirmed to preserve old-task *restricted*
accuracy strongly (93–95%) while doing comparatively little for *open*
accuracy on the oldest tasks (as low as 0.05–3.4%). It remains the right
choice for **SimpleAvg**, where full-scope KD was independently shown
(`R7/r7_simpleavg_kd_factororth_forensic_analysis.md`) to be persistently
loss-dominant (1.1–1.3× CE for the entire run, never decaying) and
historically damaging (62.98% vs. 75.16% plain). SimpleAvg's KD arms in this
script therefore stay old-seen-only.

## 10. Temperature T2 vs T4 Rationale

No prior run in this project has ever tested SimpleAvg's old-seen KD at any
temperature other than T=2. T=4 is a standard, conservative alternative
(softer teacher targets) that isolates temperature as the sole variable —
weight, warmup, scope, optimizer, and merge are all held fixed to SA-2's
values. `T²` scaling is applied generically inside the shared
`masked_kd_loss()`/trainer `compute_loss()` code via `self.kd_temperature ** 2`
— never a hardcoded per-temperature constant — so T=4 correctly produces a
16× (not a special-cased) scaling factor on the raw KD term before the
`kd_weight` multiplier is applied.

## 11. DenseOrth20

Confirmed inert for SimpleAvg across two independent runs (job 4969059, job
4970580; back-calculated loss share ≈0.04% of CE, `R7/lambda_and_kd_weight_design_audit.md`).
Kept in this script **unchanged** (SA-4) purely as a same-run control — not
because further tuning is expected to help (it is not; see the design
audit's closed verdict), but because SA-4/SA-5 anchor the DenseOrth+KD
combined-arm comparison at the new 9-epoch budget.

## 12. FactorOrth50

Confirmed, within-pipeline, to be RankExt's effective orthogonality
mechanism (+9.81pp over plain in job 4970580, vs. DenseOrth's statistically
negligible +0.07pp in job 4969059) and mechanistically well-posed for
RankExt's persistent, shared-basis adapter (unlike SimpleAvg's
independently-initialized specialists, where FactorOrth compares
arbitrary, unaligned bases — `R7/r7_simpleavg_kd_factororth_forensic_analysis.md`
Section 10). Its own formulation (λ=50, 1-epoch warmup, cumulative
frozen-block comparison) is unchanged from Section 5/6 and is not itself a
lever being tested here.

## 13. ProtectWeight30

The `RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT = 30.0` mechanism
(`compute_old_semantic_subspace`, Section 5) is a genuine third loss term,
never present in job 4969059/4970580's arms and never previously identified
as a contributor in any report before the recovery-design audit. It is
activated for the two new RankExt KD arms by adding their method names to
`RANKEXT_PROJECTED_PROTECT_METHODS` — the underlying computation is not
reimplemented, only re-enabled (confirmed identical to
`R7/vit_lora_cifar100_full5step_n5.py`'s own version by direct comparison
during Section 6's audit).

## 14. New-Block Warmup

Disabled for the same two new RankExt KD arms
(`RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS`), exactly matching Exp1's own
treatment of its two historical KD-bearing RankExt methods. Left **enabled**
for the two non-KD RankExt arms (plain, FactorOrth-alone), matching both
Exp1's and job 4970580's treatment of those arms — verified directly from
source rather than imposed as one uniform setting (task brief Section 20's
explicit instruction).

## 15. Classifier Restoration

The corrected, non-buggy restoration (`tensor[idx] = value`, a genuine
in-place `index_put_`) is used unconditionally for every RankExt arm,
including the two new full-KD arms. The historical bug is **never**
reintroduced. `R7/final_recovery_and_improvement_design_audit.md` Section 6
reasoned — from AdamW's decoupled weight-decay arithmetic, not merely
asserted — that the historical bug's most likely effect (letting protected
rows shrink via undecayed weight decay, since the "restore" step was a
no-op) would have made the exact metric it might be credited with *worse*,
not better. There is no evidence-based reason to reintroduce it, only to
test it once as a deliberate, separate ablation if resources allow (not part
of this script).

## 16. Epoch Count

Both families train for **9 epochs/CL step**, matching the historical
protocol exactly and enforced as a hard, unconditional assertion
(`assert LORA_EPOCHS == 9`, `assert RANKEXT_EPOCHS == 9` — no `FAST_RUN`
escape hatch, since this script is never run in smoke-test mode). This
removes epoch count as a confound between this run and the historical
70.72% recipe, leaving KD scope, `protect_weight`, and new-block-warmup
state as the remaining deliberate mechanism changes relative to job 4970580.

## 17. Final 9-Method Design

| # | Method | Family | KD scope | KD T | KD weight | KD warmup | Orth | λ | protect_weight | New-block warmup |
|---|---|---|---|---:|---:|---|---|---:|---:|---|
| 1 | `simple_avg` | SimpleAvg | n/a | – | 0 | n/a | none | 0 | n/a | n/a |
| 2 | `simple_avg_kd_oldseen_T2_warmup` | SimpleAvg | old_seen | 2 | 1.0 | 1 epoch | none | 0 | n/a | n/a |
| 3 | `simple_avg_kd_oldseen_T4_warmup` | SimpleAvg | old_seen | 4 | 1.0 | 1 epoch | none | 0 | n/a | n/a |
| 4 | `simple_avg_dense_orth_lam20` | SimpleAvg | n/a | – | 0 | n/a | dense_orth | 20 | n/a | n/a |
| 5 | `simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup` | SimpleAvg | old_seen | 2 | 1.0 | 1 epoch | dense_orth | 20 | n/a | n/a |
| 6 | `rank_extension` | RankExt | n/a | – | 0 | n/a | none | 0 | 0 | enabled |
| 7 | `rank_extension_fullkd_T2_protect30` | RankExt | full | 2 | 1.0 | none | none | 0 | 30 | **disabled** |
| 8 | `rank_extension_factor_orth_lam50` | RankExt | n/a | – | 0 | n/a | factor_orth | 50 | 0 | enabled |
| 9 | `rank_extension_factor_orth_lam50_fullkd_T2_protect30` | RankExt | full | 2 | 1.0 | none | factor_orth | 50 | 30 | **disabled** |

All 9 arms: seed 42, 9 epochs, arithmetic merge (SimpleAvg) / persistent
merge (RankExt), no replay, no DO-Merge, fixed classifier restoration.
Verified as the exact active method set by a static, dataset-independent
execution of the script's own configuration/assertion code (Section 17 of
the readiness report).

## 18. Risks

- **Arms 7/9 are an untested combination**: full-100-way KD, weight=1.0, no
  warmup, `protect_weight=30`, new-block-warmup-disabled, **at 9 epochs, with
  the corrected classifier restoration** — this exact configuration has never
  been run before. The historical 70.72% result used the *same* mechanism
  configuration but the *broken* restoration; job 4961962's reproduction
  confirms the mechanism is safe and reaches 70.72% under that (buggy)
  restoration. Whether the corrected restoration changes the result, and by
  how much, is genuinely unknown (Section 6 of the recovery-design audit
  argues LOW-probability-negative, not zero-probability).
- **Full-strength, unscaled combined loss** (λ=50 FactorOrth + weight=1.0
  full KD + protect_weight=30, arm 9): the codebase's own history documents
  a `COMBINED_LOSS_SCALE` mitigation introduced to avoid a full-strength
  training collapse for an analogous SimpleAvg combined arm. Job 4961962
  already proved this exact unscaled combination is safe for RankExt
  (it produced 70.72%, not a collapse) — but that was under 6 fewer... no,
  under the *same* 9 epochs and the *buggy* restoration. This script keeps
  the combination unscaled by design (matching the task brief's explicit
  values), consistent with the evidence that it is not fragile.
- **SA-3 (T=4) has no precedent** — a real chance it lands worse, better, or
  between SA-2 and plain; no specific outcome is predicted with confidence.
- **18-hour walltime is an estimate**, not a measurement — no prior run at
  this exact epoch count/arm mix has been timed end-to-end.
- **Cross-run comparisons in the produced summary tables remain confounded**
  (epochs, KD scope, and `protect_weight` all differ simultaneously in most
  rows vs. job 4970580) — the script labels every such row explicitly; only
  the within-run comparisons (Section 7 of its reporting output) are strong
  evidence.

## 19. Final Recommendation

Implementation is complete and statically validated (see
`R7/final_9method_5x20_static_readiness.md`). This design directly executes
the highest-probability, lowest-unnecessary-risk path identified by the
prior recovery audit: it isolates KD scope, `protect_weight`, and
new-block-warmup state as a bundle against the corrected classifier
restoration (the one mechanism deliberately *not* reverted), at the
historical epoch count, using code that was already present and verified
rather than newly written. **Recommendation: READY FOR USER-MANAGED GIT PUSH
AND CLUSTER RUN** — no further design changes are indicated before that step.
