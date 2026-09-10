# Supervisor CIFAR-100 Followup Audit

**Date:** 2026-09-10/11 (2026-09-11 addendum: `old/` notebook cross-check, see bottom of this
report). **Status: AUDIT COMPLETE. Two experiments prepared, NEITHER launched.**
ImageNet-100 generalization study status: **POSTPONED** (see
`thesis_agent/reports/imagenet100_generalization_preparation.md`, updated).

This report resolves the supervisor's feedback on the earlier CIFAR-100 4-step results (quoted in
full below) against every existing R6/historical experiment, before any new job is prepared or run.

> "Did you run the 5x20 setting again with the fixed rank ext? [...] i think the 5x20 setting with
> fixed rank ext would be helpful to compare exactly, what changed now. Apart from that, it could be
> interesting to test more extreme scenarios, e.g. 10x10 and 2x50. Also [...] more classes in step 0
> [...] To summarize [...] rerun the 5x20 scenario with fixed rank extension, same rank as before
> the fix. run 5x20 with rank 32 for simple avg, and rank 32,64,... for rankext (fixed version ofc)
> [...] after that, i would look a bit, what exactly happens with the rank ext"

---

## PHASE 1 — Restoration to canonical CIFAR-100 (DONE)

`vit_lora_cifar100_full5step_n5.py` was surgically restored. The full diff is `git diff` against
the pre-restoration working tree; summary: **396 lines removed, 26 added**, all confined to the
ImageNet-100 preparation (commits `944aa0f`, `2f69291`). Removed (category B — introduced only for
ImageNet-100):

- `N5StopAfterSetup` exception class and the `N5_SKIP_TRAINING_DRIVER` setup-only import guard
  (existed solely to support `scripts/smoke_test_imagenet100.py`).
- `DATASET_REGISTRY`, `N5_DATASET_NAME`/`N5_EXPERIMENT_LABEL` env-var plumbing, `IMAGENET_ROOT`.
- `IMAGENET100_SYNSETS`, `IMAGENET100_CLASS_ORDER`, `IMAGENET100_SYNSETS_CMC_ALTERNATIVE`,
  `IMAGENET100_WNID_TO_LABEL` and their validity asserts.
- The `imagefolder`/local-ImageNet-1k dataset-loading branch and the ImageNet-100 label-remap block
  in the main dataset-load cell.
- The `DATASET_REGISTRY["final_eval_split"]` indirection in `make_eval_dataset` (restored to the
  literal `dataset["test"]`).
- The consolidated "EXPERIMENT-INVARIANT ASSERTIONS" block added for the ImageNet prep. This one is
  **not** re-added even in a CIFAR-only form: it hardcoded `NUM_STEPS==4`/`CLASSES_PER_STEP==25`,
  which would fail-fast on exactly the 5x20 reruns this followup needs to prepare (Phase 3) — it was
  a guard against a problem specific to the (now-removed) dataset-switching mechanism, not a
  generally-useful invariant this project's own protocol-depth methodology (5x20/20x5/4x25, all
  legitimate) should have.

Preserved (category A — pre-existed ImageNet or is independently useful, per the task's explicit
list): the corrected/fixed RankExt implementation (every post-2026-07-21 calibration/warmup/
feature-protection/eval fix), `SEED` default 42 with the independently-useful `REPLICATION_SEED`
env-var override (this override predates the ImageNet work — added 2026-09-06 for the canonical
seed-123/2026 replication plan, unrelated to ImageNet), R6-16 capacity-sensitivity history, R6-17/
seed123 records, Chapter 2/3 work, bibliography work, all R6 result archives, and all Agent
evidence/history. `USE_RANKEXT_RANK_SCHEDULE_WIDE` was already `False` at the pre-ImageNet commit
(`0315ec5`) — the `[[widerank-capacity-experiment]]` memory note describing it as "still live" was
already stale before this session; verified directly against the live file, not trusted from memory.

**Safety fix (not requested verbatim, but required by "do not launch anything"):** with the setup
guard removed, `scripts/smoke_test_imagenet100.py` would have silently fallen through into a REAL
full 8-method training run if executed (it depends on the now-removed
`N5StopAfterSetup`/`N5_SKIP_TRAINING_DRIVER` mechanism to stop early). A guard was added at the top
of that script so it refuses to run and explains why, instead of launching training. Preserved,
not deleted — `scripts/verify_imagenet100_local.py` is self-contained and unaffected.

**Validation performed:**

| Check | Result |
|---|---|
| `py_compile` on the restored script | PASS |
| `dataset = load_dataset(...)` call sites | exactly one, unconditional, `"cifar100"` |
| `grep -i imagenet` on the script | 0 hits outside one explanatory historical comment |
| `SEED` default | `42` (`REPLICATION_SEED` env-var override, pre-existing/independent of ImageNet) |
| `NUM_STEPS` / `CLASSES_PER_STEP` | `4` / `25` (canonical 4x25) |
| `RANKEXT_RANK_SCHEDULE` | `[20, 40, 60, 80]` (canonical, fixed implementation) |
| `USE_RANKEXT_RANK_SCHEDULE_WIDE` | `False` |
| Result files deleted by this session | **0** (see "Pre-existing condition" below) |

**Pre-existing condition flagged, not touched:** at session start, `git status` already showed 136
tracked files (in `analysis_R3/`, `analysis_R4/`, `analysis_pipeline_audit/`, `analysis_rankext_*/`,
`analysis_recency_fix*/`, `analysis_revert_run/`, `analysis_simple_avg_overfit/`,
`analysis_strict_audit/`, and more) deleted from the working tree but **not committed** — this
predates this session and was not caused by it. Per "do not delete historical evidence," this audit
did not commit that deletion or run anything (`git add -A`, `git clean`, etc.) that would finalize
it; the files remain fully recoverable via `git checkout HEAD -- <path>` since they are only
working-tree-deleted, still tracked in git history. **This is flagged for the author's own decision
(intentional cleanup vs. accidental) — not resolved by this audit**, since it is unrelated to the
ImageNet restoration or the supervisor's CIFAR-100 followup.

---

## PHASE 2 — Supervisor Request Matrix

Recovered from `thesis_agent/sources/R6_catalog.md` (which itself audited actual `run_config.json`/
`hyperparameters_by_method.json`/result tables, not folder names) and directly from
`vit_lora_cifar100_full5step_n5.py`'s git history.

| # | Supervisor request | Exact required configuration | Existing candidate | Exact match? | Result available? | Still needed? | Reason |
|---|---|---|---|---|---|---|---|
| 1 | Fixed-RankExt 5x20 rerun, same rank as pre-fix | 5x20, RankExt `[16,32,48,64,80]`, all 8 methods, fixed RankExt | Pre-fix: `R6/results_fix2_20260721_light` \| `results_4904629_light` (job 4904629, 20260817, all 8 methods, schedule `[16,32,48,64,80]`). Post-fix candidates: `results_4906414_light`…`results_4917775_light` (jobs 4906414–4917775) | **NO** | Partial | **YES — MISSING** | The post-fix 5x20 runs (#3–#11 in the catalog) all disabled `simple_avg` to isolate RankExt tuning (calib floor → gamma → cosine-classifier eval → NCM eval → feature protection → weight → lambda sweep → nullspace variant). No run has BOTH all 8 methods AND the fully-fixed RankExt. The project moved to the 4x25 protocol immediately after the RankExt-only fix sequence closed (`8153008`→`eb30de3`→`1ed6279`), without ever returning to 5x20 with all 8 methods. |
| 2 | 5x20, SimpleAvg r32, RankExt `[32,64,96,128,160]` (fixed) | 5x20, LORA_R=32/alpha=64, RankExt `[32,64,96,128,160]`, fixed RankExt, all 8 methods | `076a1b5` (2026-07-21, "parity capacity test"): 5x20, RankExt schedule **exactly** `[32,64,96,128,160]` | **NO** | Partial (superseded) | **YES — MISSING** | Two disqualifying differences: (a) SimpleAvg was left at **rank 80** in that run (the "give RankExt more total params than SimpleAvg" asymmetric-capacity test) — SimpleAvg was never run at rank 32; (b) that run used the **pre-fix** RankExt implementation (predates the whole 2026-07-21→08-23 fix sequence: cosine classifier, NCM eval, feature protection, family-aware/confidence-weighted calibration, new-block warmup). It was reverted the same session specifically so the *next* run could isolate FIX 1 as the only lever. Not the supervisor's exact request under any reading. |
| 3 | R6-16 capacity test (context, not a direct request) | 4x25, RankExt `[40,80,120,160]` | `R6/rankext_widerank40_4933319` | N/A (different protocol/schedule/purpose) | YES | N/A — see Q-C | Directly relevant evidence, not a substitute for #1/#2 (different protocol: 4x25 not 5x20; different schedule; SimpleAvg unchanged at 80, not 32). |
| 4 | Best-epoch selection: "leave it as-is" | (no config — a validation of existing behavior) | `USE_BEST_EPOCH_SELECTION` / `EpochValidationCallback` in n5.py | YES (verified against live code) | N/A | **NO — RESOLVED** | See Question E below. |
| 5 | 10x10 | 10 steps x 10 classes | none | — | — | **OPTIONAL** | Supervisor's own final summary lists only #1/#2 as the two things to run; 10x10/2x50 were framed as "could be interesting," not committed to. |
| 6 | 2x50 | 2 steps x 50 classes | none | — | — | **OPTIONAL** | Same as above. |
| 7 | 50+5x10 (more classes in step 0) | 1 step of 50 + 5 steps of 10 | none | — | — | **OPTIONAL** | Same as above — offered as "a thing you could try," not in the final numbered summary. |
| 8 | Different dataset | (ImageNet-100 or other) | ImageNet-100 preparation (`thesis_agent/reports/imagenet100_generalization_preparation.md`) | — | NO (not launched) | **AFTER CIFAR FOLLOWUP** | Supervisor: "i would test a different dataset once we found a clear favorite method on this one" — explicitly gated on finishing the CIFAR-100 followup first. Now formally POSTPONED, not abandoned. |

---

## Question A — Was the fixed 5×20 rerun already run?

1. **Pre-fix 5x20 experiment existed?** YES. `results_fix2_20260721_light` (duplicate of
   `R3/results_fix2_20260721_light`) and its near-identical follow-up `results_4904629_light`
   (job 4904629, 20260817) — both all-8-methods, 5x20, epochs=9, RankExt schedule `[16,32,48,64,80]`
   (recovered from `hyperparameters_by_method.json` via `R6_catalog.md`, not guessed).
2. **Its RankExt schedule:** `[16, 32, 48, 64, 80]` (final cumulative rank 80, matching SimpleAvg's
   rank 80 — parity, unlike Experiment 2's deliberate asymmetry).
3. **What was wrong pre-fix (from `R6_catalog.md` §5, `analysis_recency_fix*`):** two compounding
   issues, both since fixed: (a) classifier-row-norm **recency bias** in the open 100-way argmax —
   frozen old-class rows lose the scale competition against newly-trained rows even when their
   *representation* is fine (later fixed via family-aware/confidence-weighted regime-grouped
   calibration for `rank_extension`, then made consistent across families for R6-15); (b)
   insufficient new-block training signal / warmup interference for freshly-added RankExt blocks
   (later fixed via new-block output warmup, feature-anchor-based first-step fix, and the eventual
   removal of the feature-anchor lever in favor of a pretrained-backbone anchor). The wide-rank
   `[32,64,96,128,160]` capacity test (`076a1b5`) was tried and **reverted** in the middle of this
   debugging sequence — capacity was tested and rejected as *the* explanation before the real fixes
   (calibration, warmup) were found.
4. **Post-fix 5x20 rerun with the SAME rank config, all methods?** **NO.** Every 5x20 run after the
   fix sequence began ran RankExt-only (`simple_avg` disabled, per `R6_catalog.md` run notes #3–#11).
   The last all-8-methods 5x20 run (#2, job 4904629) predates nearly the entire fix sequence
   (calibration floor/gamma tuning, cosine-classifier eval, NCM eval, feature protection, nullspace
   variant all came after it, jobs 4906414→4917775). Immediately after the fix sequence closed, the
   project moved to the 4x25 protocol (`eb30de3`, `1ed6279`) rather than returning to 5x20.
5. **All relevant methods?** N/A — no candidate run exists to check.
6. **Scientifically comparable pre-fix vs. post-fix?** **Cannot currently be produced** — the
   comparison the supervisor is asking for genuinely does not exist yet. **→ Experiment 1, MISSING.**

## Question B — Was the supervisor's rank-32 5×20 test already run?

**NO**, under a strict, field-by-field reading. The closest historical candidate, `076a1b5`'s
"parity capacity test" (5x20, RankExt `[32,64,96,128,160]`), differs on both axes that matter:
SimpleAvg stayed at rank 80 (not 32 — that run's whole point was to give RankExt *more* capacity
than SimpleAvg, not to reduce SimpleAvg to match a smaller shared budget), and it used the pre-fix
RankExt implementation the supervisor explicitly wants excluded ("fixed version ofc"). R6-16
(4x25, `[40,80,120,160]`) is a different protocol, a different schedule, and again leaves SimpleAvg
at rank 80. **→ Experiment 2, MISSING.**

## Question C — What has R6-16 already told us about capacity?

**R6-16 answers:** within a 4x25/fixed-RankExt/rank-80-SimpleAvg setting, doubling RankExt's
cumulative rank (80→160, schedule `[40,80,120,160]`) produced only +0.6 to +1.6pp all_seen-accuracy
gains across the four RankExt variants, first-step accuracy got *worse* for all four despite double
the step-1 rank, and the ~30–40pp non-KD SimpleAvg gap closed by under 5%. Locked interpretation
(verbatim, do not restate stronger): **"the tested increase in RankExt capacity does not appear to
be the dominant limiting factor within the evaluated range."**

**R6-16 does NOT answer:** (a) whether this holds under 5x20 (a materially harder/different protocol
— per `R6_catalog.md`'s own cross-run observation, 5x20 RankExt+KD+FactorOrth sits ~69–70% vs.
4x25's ~73–74%, so the two protocols are not interchangeable evidence); (b) the supervisor's
specific, much *larger* deliberate-asymmetry design — R6-16 keeps SimpleAvg fixed at rank 80 while
doubling RankExt to 160 (2x asymmetry); the supervisor's request is SimpleAvg at rank 32 vs. RankExt
at 160 (5x asymmetry) — a qualitatively bigger "clear advantage" than anything tested so far; (c)
whether RankExt can *exploit* that capacity when the comparison partner (SimpleAvg) is also
resource-constrained, rather than fixed at its own historical maximum. **R6-16 does not replace
Experiment 2** — it is directionally consistent prior evidence, not a substitute measurement.

## Question D — SimpleAvg capacity: correcting the supervisor's assumption

Verified directly from executable source (`vit_lora_cifar100_full5step_n5.py`,
`simple_average_deltas()` and its call site around `run_simple_avg_variant()`), not from a
memory/summary:

- Each incremental step trains a **fresh, independently initialized** rank-r LoRA adapter (`A_t`,
  `B_t`) on top of the **same frozen base** — steps are not sequentially merged into one running
  LoRA factorization.
- Each step's contribution is materialized as a **dense per-step update**
  `Δ_t = scaling · (B_t @ A_t)` (`state["deltas"][key] = scaling * (B @ A)`, one dense matrix per
  target-module per step).
- `simple_average_deltas()` merges all `N` steps' dense updates by **arithmetic mean**:
  `merged = torch.stack([Δ_1, ..., Δ_N]).mean(dim=0)` — a mean of dense matrices, **not** a merge
  of the low-rank `A`/`B` factors into one rank-r pair.
- Consequence: the final merged update is **not structurally constrained to rank r**. With `N`
  independent rank-r contributions, the dense mean's rank is bounded above by `min(N·r, layer
  dims)` before any cancellation/linear-dependence between steps' updates (not asserted or measured
  here, so treat the bound as an upper bound, not a claim that the effective rank reaches it).

**Supervisor's assumption does NOT match the code.** The supervisor recalled "one lora into which
subsequent loras are merged sequentially... the rank would basically stay at 80 the whole time." The
actual mechanism keeps each step's `r`-rank factorization separate through training, then averages
the resulting *dense* matrices — so the theoretical dense-rank ceiling scales with the number of
steps, not just `r`. For canonical 4x25/r=80: ceiling ≤ 320. For the requested 5x20/r=32: ceiling ≤
160. **The configured LoRA rank in the rank-32 experiment remains 32** (that is what is actually
trained, and what should be reported as "SimpleAvg rank" in any config table) — do not call it
"rank 160"; that number describes only a loose upper bound on the *merged dense update's* rank, not
the trained adapter's rank, and is not adjusted for cancellation between steps.

## Question E — Best-epoch / validation selection

Verified directly from source: `USE_BEST_EPOCH_SELECTION = True`; `EpochValidationCallback`
computes pure validation CE via `compute_dataset_ce_loss()` once per epoch (bypassing
`compute_loss()`'s regularized total, which for lambda_orth/KD methods is NOT pure CE — this was
itself a prior, already-fixed bug, "PRE-THESIS FIX 1"), keeps an in-memory snapshot of the
trainable parameters whenever validation CE improves, and `train_with_trainer()` reloads that
snapshot after `trainer.train()` finishes. This is exactly "lowest validation CE within this CL
step," matching the supervisor's own description ("it seems to quite accurately find the lowest
validation loss each time"). **No bug found. RESOLVED / NO ACTION** — no new stabilization
experiment, no smoothing, no best-epoch logic change.

## Question F — Optional protocols (10x10 / 2x50 / 50+5x10)

Classified **OPTIONAL / SECONDARY**, not mandatory: the supervisor's own final, explicit numbered
summary ("To summarize, i think the following experiments would be useful... after that, i would
look a bit...") lists only the two items in the matrix above. 10x10, 2x50, and "50+5x10" were each
introduced with hedged language ("could be interesting," "a thing you could try") and are absent
from that closing summary. **Not implemented in Phase 3** per the explicit instruction not to
auto-prepare them; no evidence found of a later, more explicit request for any of the three.

---

## Scientific interpretation (what the supervisor is diagnosing)

1. **Why does plain RankExt perform much worse than SimpleAvg?** Historically, three compounding
   causes were found and fixed one at a time (not a single root cause): classifier-row recency bias
   in the open 100-way argmax (masked representation quality behind a scale artifact — the
   "restricted vs. open" accuracy gap, `analysis_pipeline_audit`), insufficient/interfered new-block
   training signal (new-block warmup, later a pretrained-backbone anchor for the first step), and —
   tested and *rejected* — raw representational capacity (R6-16, R6/`076a1b5`'s reverted test). After
   fixes, plain (non-KD) RankExt variants still never exceed ~42% all_seen accuracy in any 5x20 run,
   so residual underperformance vs. SimpleAvg on the non-KD side is not yet fully explained — the
   supervisor's #2 request is designed to gather more evidence on this specific point.
2. **Why does KD strongly help RankExt?** Every 5x20 diagnostic run shows RankExt+KD variants
   30+pp above their non-KD counterparts; KD is confirmed "essential for RankExt-family methods to
   be competitive" (`R6_catalog.md` cross-run observation #2) — consistent with KD supplying a
   stable target signal a growing/frozen-block architecture otherwise lacks turn-to-turn.
3. **Why do KD/FactorOrth affect SimpleAvg differently (sometimes negatively)?** Not fully isolated
   by any single existing run; this is one of the still-open questions the supervisor's own
   diagnostic instinct ("after that, i would look a bit, what exactly happens with the rank ext") is
   aimed at, alongside the RankExt-side mechanism. See the RankExt internal-diagnostic plan below —
   this can be investigated from ALREADY-SAVED artifacts (per-method loss decompositions,
   `analysis_revert_run`'s merge-mechanism diagnostics) without a new run.
4. **Is RankExt's poor performance primarily a capacity problem?** R6-16 says: not dominantly, within
   the 2x-asymmetry range it tested (Q-C). Experiment 2 (5x asymmetry, resource-constrained
   SimpleAvg) is the next, more decisive test of this question, and does not yet exist.
5. **Does protocol depth change the behavior?** YES, substantially: RankExt+KD+FactorOrth ≈ 69–70%
   under 5x20, ≈73–74% under 4x25, and collapses to ≈26% under 20x5 (`R6_catalog.md` cross-run
   observation #1) — more/smaller CL steps is drastically harder for this model/method combination.
   This is exactly why Experiment 1/2 must be run under 5x20 specifically, not read off the 4x25
   R6-15/R6-16 evidence.
6. **Can RankExt exploit a deliberate capacity advantage?** Untested at the magnitude (5x) the
   supervisor proposed — R6-16 tested only a 2x asymmetry and found little effect; Experiment 2 is
   the first test of the larger, resource-constrained-competitor version of this question.

---

## PHASE 3 — Experiments prepared (NOT launched)

**2 missing mandatory experiments identified; both prepared, neither submitted.**

All artifacts are under `experiments_prepared/` (see that directory's own `README.md`). Each
`.py` is a dedicated copy of the just-restored canonical `vit_lora_cifar100_full5step_n5.py` with
only the constants below changed — no SimpleAvg/RankExt/KD/FactorOrth/calibration/metric algorithm
touched anywhere.

### Experiment 1 — fixed-RankExt 5x20 rerun

`experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py` /
`experiments_prepared/slurm/exp1_5x20_fixed_rankext_seed42.sbatch`

| Setting | Value |
|---|---|
| Dataset | CIFAR-100 |
| Protocol | 5x20 (`NUM_STEPS=5`, `CLASSES_PER_STEP=20`) |
| Seed | 42 |
| Backbone | CLIP ViT-B/16 |
| SimpleAvg rank / alpha / scaling | 80 / 160 / 2.0 (unchanged from pre-fix reference) |
| RankExt schedule / scaling | `[16, 32, 48, 64, 80]` (same as pre-fix reference) / 2.0 |
| KD | weight=1.0, T=2.0, step2+, no warmup (canonical, untouched) |
| FactorOrth | lambda=50 (canonical, untouched) |
| Calibration | current canonical grouped policy, consistent across families (no R6-14 asymmetry) |
| Epochs | 9 (matching every historical 5x20 job; canonical 4x25 uses 7) |
| Methods | all 8 |
| Run name | `clip_vit_lora_cifar100_5x20_fixed_rankext_same_rank_as_prefix_reference_seed42` |

### Experiment 2 — rank-32 SimpleAvg vs. RankExt `[32,64,96,128,160]` capacity test

`experiments_prepared/vit_lora_cifar100_5x20_rank32_capacity_seed42_EXP2.py` /
`experiments_prepared/slurm/exp2_5x20_rank32_capacity_seed42.sbatch`

| Setting | Value |
|---|---|
| Dataset | CIFAR-100 |
| Protocol | 5x20 |
| Seed | 42 |
| Backbone | CLIP ViT-B/16 |
| SimpleAvg configured rank | **32** |
| SimpleAvg alpha | **64** (`2 * LORA_R`, the same formula used everywhere else in this file) |
| SimpleAvg effective scaling | **2.0** (preserved — verified against `add_lora`'s actual PEFT config, not assumed) |
| RankExt cumulative schedule | **`[32, 64, 96, 128, 160]`** (exact supervisor request) |
| RankExt effective scaling | **2.0** (`RANKEXT_ALPHA_PER_RANK`, constant at every step by construction) |
| KD | weight=1.0, T=2.0 (canonical, untouched) |
| FactorOrth | lambda=50 (canonical, untouched) |
| Calibration | current canonical grouped policy, consistent across families |
| Epochs | 9 (matching every historical 5x20 job, and paired with Experiment 1) |
| Method set | **all 8** — see justification below |
| Run name | `clip_vit_lora_cifar100_5x20_rank32_capacity_fixed_rankext_seed42` |

**Method-set decision (8, not a minimal subset):** the supervisor's stated puzzle is specifically
that KD/FactorOrth affect the two families in *opposite* ways ("strange that the rank extension
seems to benefit from KD and orth., while the simple average suffers from both of them") — that
observation requires all four SimpleAvg variants and all four RankExt variants in the same run to
even be visible. Dropping to a minimal subset (e.g. plain + one combined variant per family) would
save roughly half the wall-clock but would foreclose exactly the cross-family comparison the
supervisor is asking about. Nothing in the historical record suggests compute pressure has ever
been the constraint for a single 5x20 job (all prior 5x20 diagnostic jobs ran on a single
`gpu:a40:1` allocation). **Decision: all 8 methods for both experiments.**

**Rank-structure / no-cross-contamination assertions:** both files assert, at import time before
any training: `SEED==42`; `NUM_STEPS`/`CLASSES_PER_STEP` match the intended protocol;
`USE_RANKEXT_RANK_SCHEDULE_WIDE is False`; the active RankExt schedule equals the intended one
exactly; `LORA_R`/`LORA_ALPHA` (and, for Experiment 2, effective scaling for BOTH families
independently, since the old rank-parity assert is deliberately broken there — see that file's own
comment); `LAMBDA_ORTH==50.0`; `KD_WEIGHT==1.0`/`KD_TEMPERATURES==[2.0]`; the enabled method set
equals the canonical 8; and `RUN_NAME_BASE` contains no stale/wrong tokens (`imagenet`, `4x25`).
Each file also prints a startup diagnostic block (dataset, protocol, seed, method set, both
families' rank/alpha/scaling, KD config, FactorOrth lambda, epochs, calibration policy, run name)
before training starts.

**SLURM:** both `.sbatch` files request `--gres=gpu:a40:1 --partition=allgroups`, include a
pre-flight `grep` sanity check on the target `.py` (protocol/rank/seed) that fails before Python
even starts, and use a `--time=14:00:00` budget (canonical 4x25/7-epoch/8-method run measures
~5:45 wall-clock, directly, per `thesis_agent/reports/canonical_3seed_runtime_plan.md`; 5x20/9-epoch
is conservatively budgeted higher and should be tightened once a real wall-clock is observed).
**Neither script has been submitted.**

---

## RankExt internal-diagnostic plan (per "after that, i would look a bit, what exactly happens with
## the rank ext" — no new experiment proposed for this yet)

Artifacts **already saved** by every historical RankExt run (no rerun needed to start this
analysis): per-step open AND restricted accuracy
(`per_step_accuracy_open_vs_restricted_by_method.csv` — directly isolates recency-bias-style scale
artifacts from genuine representation quality); old/new/all_seen accuracy and BWT/forgetting per
method; per-epoch train and validation CE (`training_loss_history_by_epoch.csv`); best-epoch
selection records (`best_epoch_selected_by_method_step.csv`); classifier row-norm calibration
diagnostics (`classifier_row_norm_*`); LoRA block/factor norms and merge-mechanism logging
(`analysis_revert_run`'s comment-1 diagnostics — built specifically to test whether later steps'
orthogonality-constrained deltas dilute step 1's contribution once `simple_average_deltas()`
averages them); KD loss and FactorOrth loss decomposition per method
(`train_kd_loss_*`/`train_factor_orth_loss_*` columns, `loss_summary_by_method_df`); rank-structure
tables per step (`rank_extension_step_N_rank_structure.csv`).

**Sufficient for a first pass without any new run:** yes — the open-vs-restricted split, the loss
decomposition, and the merge-mechanism diagnostics together can already characterize *where* RankExt
loses to SimpleAvg (representation vs. classifier-scale vs. dilution-on-merge) for every existing
5x20/4x25 run, including a future Experiment 1/2 once those exist. **New instrumentation is not
proposed at this time** — only propose it if this first pass, once actually performed, finds a
question these artifacts cannot answer.

---

## Agent update

- `thesis_agent/reports/imagenet100_generalization_preparation.md` — status banner added:
  **POSTPONED**, not abandoned; documents exactly what was removed from the active script and why,
  and that all of it is recoverable from git history if the study resumes.
- `thesis_agent/CONTINUE_LATER.md` — section 7 updated to POSTPONED; "NEXT TASK" section
  repointed at this followup (was ambiguously listing the ImageNet smoke test as a live option).
- `scripts/smoke_test_imagenet100.py` — guarded (refuses to run) rather than left as a
  now-broken script that would silently launch full training if executed.
- This report (`thesis_agent/reports/supervisor_cifar_followup_audit.md`) — new.
- **Not modified:** thesis chapters (Chapter 2/3/4/5/6), `data/*.jsonl` (the generated
  knowledge-base files — per this project's own convention, edit `build_knowledge_base.py` and
  re-run it, not the generated files by hand; doing so is a reasonable near-term follow-up but was
  out of scope for this restoration/audit pass and risks side effects without being able to execute
  and verify the build/validate scripts in this pass).

---

## ADDENDUM (2026-09-11) — Old notebook cross-check

**File identified:** `old/vit_lora_cifar100_full5step_n5.py` — the only file under `old/`. It is
itself a converted-notebook `.py` (same convention as the active `vit_lora_cifar100_full5step_n5.py`
— per this project's own established pattern, these `.py` files ARE the "notebooks," executed
top-to-bottom; there is no separate `.ipynb`). Filesystem mtime **2026-08-25**.

### What this file actually is (important correction of the working assumption)

**This is NOT a pre-fix 5x20 snapshot.** Its live configuration is `NUM_STEPS=4`,
`CLASSES_PER_STEP=25`, `RANKEXT_RANK_SCHEDULE=[20,40,60,80]`,
`RUN_NAME_BASE="clip_vit_lora_cifar100_4x25_final_8methods_thesis_comparison"` — i.e. it is a
**4x25-protocol, already-fixed-RankExt, R6-15-era snapshot** (matches the "Prepare final fair 4x25
eight-method thesis run" / "Run 4x25 full-strength KD orth" commits, both dated 2026-08-25). A
line-by-line `diff` against the current restored canonical `vit_lora_cifar100_full5step_n5.py`
shows exactly **three** differences, all cosmetic/non-scientific: `SEED = 42` (literal) vs.
`SEED = int(os.environ.get("REPLICATION_SEED", "42"))` (env-overridable, same effective value),
the literal `RUN_NAME_BASE` string, and the absence of the (correctly, separately removed)
ImageNet-100/seed-123-replication plumbing added after 2026-08-25. Every SimpleAvg/RankExt/KD/
FactorOrth/calibration/warmup constant is byte-identical to current canonical. **This file
corroborates that current n5.py's methodology is unchanged since the R6-15 era; it cannot, by
itself, recover the pre-fix 5x20 configuration**, because it postdates the entire 2026-07-21→08-23
RankExt fix sequence and the 2026-08-24 protocol switch to 4x25.

It DOES contain useful **historical documentation embedded in comments** (not live config) about
the 5x20 era, used below as a third, independent corroborating source alongside `R6_catalog.md`
and job 4904629's own artifacts.

### Provenance table

| Setting | Value used for Experiment 1 | Source(s) | Agreement/conflict |
|---|---|---|---|
| 5x20 RankExt rank schedule | `[16, 32, 48, 64, 80]` | (1) `R6_catalog.md`, read directly from job 4904629's `hyperparameters_by_method.json` field `"lora_rank_schedule (rankext)"`; (2) `old/...py` line ~1137-1158 comment: "5x20's schedule was [16,32,48,64,80] -- +16 rank per +20-class step, i.e. 0.8 rank units per class"; (3) `old/...py` line ~1213 comment: "Reverted to the default (narrow, [16,32,48,64,80]) schedule" | **AGREE** (3 independent sources) |
| 5x20 "settled flagship" per-step numbers | `[63.80, 69.40, 63.10, 73.50, 83.30]`, mean **70.62** | `old/...py` line ~8392 comment (embedded reference value, not live config) | **Cross-checks exactly** against `R6_catalog.md`'s reported all_seen accuracy for job 4914807's `rank_extension_orth_factor_lam_50_kd_T2` (**70.62%**, "best combined-method accuracy in the whole 5x20 job-ID sequence") — same number from two independently-authored sources |
| SimpleAvg configured rank / alpha (Experiment 1) | 80 / 160 (unchanged from pre-fix reference) | `R6_catalog.md` (job 4904629 config) and current n5.py | **AGREE** |
| Epochs (5x20) | 9 | `R6_catalog.md` (job 4904629: `epochs=9`) | **AGREE** — not independently re-derivable from `old/...py`, which is a 4x25/epochs=7 snapshot |
| KD (weight/T) | 1.0 / 2.0 | `R6_catalog.md`, `old/...py`, current n5.py | **AGREE**, all three |
| FactorOrth lambda | 50 | `R6_catalog.md`, `old/...py`, current n5.py | **AGREE**, all three |
| Seed | 42 | `R6_catalog.md`, `old/...py`, current n5.py | **AGREE**, all three |
| LR (SimpleAvg / RankExt) | 5e-5 / 1e-4 | `R6_catalog.md` (job 4904629), `old/...py` | **AGREE** |
| **SimpleAvg target modules** | **CURRENT (2: q_proj/v_proj)**, NOT job 4904629's own config | `R6_catalog.md`: job 4904629 (and its config template) ran SimpleAvg with **4 modules** (q,k,v,out_proj) and "global" calibration. `old/...py` lines ~370-428: documents the SAME setting going 4-module (BASELINE) → stays 4-module through a 2026-07-16 RankExt-only revert → briefly narrowed to 2-module in a same-day "STRICT-FAIRNESS REDESIGN" → briefly restored to 4-module → **FINAL correction, 2026-08-25: set to 2-module (q,v) for all 4 simple_avg methods, explicit instruction**. Current n5.py: 2-module (unchanged since). | **CONFLICT, resolved explicitly (not silently):** job 4904629's SimpleAvg config (4-module/"global" calibration) is superseded, later-project-history, by a deliberate 2026-08-25 fairness fix. Reproducing the pre-fix reference's SimpleAvg config verbatim would **reintroduce the R6-14-style calibration/target-module asymmetry** the original audit was explicitly told to avoid. **Decision: use CURRENT (2-module, consistent calibration) for both families in Experiment 1** — this is "the current corrected methodology" the task said to keep, not "historical rank/protocol" (which is limited to protocol depth, RankExt's own rank schedule, and epoch count). Documented in-line in `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`. |
| **Calibration algorithm (both families)** | **CURRENT (`confidence_weighted_regime_grouped`, both families)**, NOT job 4904629's own config | `R6_catalog.md`: job 4904629 used `calib=global` for simple_avg, `calib=confidence_weighted_regime_grouped` for rank_extension (asymmetric). `old/...py` lines ~683-723: documents the 2026-08-25 "CALIBRATION EXPERIMENT" that switched simple_avg from `global` to `confidence_weighted_regime_grouped` specifically so both families use the SAME algorithm. Current n5.py: same, unchanged since. | **Same conflict/resolution as target modules above** — this asymmetry is exactly what the original audit instructed be avoided ("Do not reintroduce the R6-14 calibration asymmetry"; job 4904629 predates the fix that resolved it, so its own calibration split is the asymmetry, not R6-14's specific variant of it, but the same category of problem). **Decision: current, consistent calibration for both families.** |

### Verdict

**Reconciled across all three required sources** (old notebook, job 4904629 artifacts, current
fixed n5.py). No unresolved conflicts remain: the two genuine discrepancies found (SimpleAvg target
modules and calibration algorithm) are not contradictions between sources — all three sources agree
on the *history* (job 4904629 used the older/asymmetric config; that config was deliberately
changed on 2026-08-25; current n5.py has the new config) — the only judgment call was which era's
SimpleAvg config Experiment 1 should use, and that call is now explicit, documented at three
locations (this table, `EXP1.py`'s own provenance comment, `EXP2.py`'s own provenance comment)
rather than silently defaulted. **Experiment 1, as prepared, is a faithful "fixed RankExt, same
RankExt rank schedule as before the fix" rerun**: RankExt's own configuration (schedule, warmup,
calibration, KD, FactorOrth) is fully reconciled and matches the pre-fix reference's rank exactly
while using the current fixed implementation; SimpleAvg's configuration deliberately uses the
current (not pre-fix) target-module/calibration settings, for the explicit, documented reason
above — not because those two settings were overlooked. No changes to the already-prepared
`experiments_prepared/` files were required beyond adding this documentation; their actual
constants were already correct against this reconciliation.

## Unresolved scientific questions (carried forward, not resolved by this audit)

- Why KD/FactorOrth help RankExt but hurt SimpleAvg specifically (mechanism, not just the
  observation) — partially addressable from existing artifacts per the diagnostic plan above,
  not yet actually performed.
- Whether RankExt's non-KD variants' ~35–42% all_seen ceiling (5x20, fixed implementation) has a
  cause beyond calibration/warmup/capacity — the three causes tested and fixed/rejected so far may
  not be exhaustive.
- Results of Experiment 1 and Experiment 2 themselves — genuinely unknown until run; this report
  makes no claim about their outcome.
