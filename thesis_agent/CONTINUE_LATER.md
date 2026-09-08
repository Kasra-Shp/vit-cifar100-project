# Build checkpoint — 2026-08-29 (+ 2026-09-01 Chapter 3 prep, + 2026-09-05 R6-16 sync)

Read this file first before resuming any work on `thesis_agent/`.

## 2026-09-05 — FINAL Chapter 3 sync: R6-16 (wide-rank capacity-sensitivity control) integrated (COMPLETE)

A new RankExt capacity-sensitivity experiment, **R6-16** (job 4933319,
`R6/rankext_widerank40_4933319/...`, RankExt cumulative rank schedule widened
`[20,40,60,80]`→`[40,80,120,160]`, final cumulative rank 80→160, effective
LoRA scaling held fixed at 2.0, SimpleAvg completely unchanged), was run,
fully analyzed (`thesis_agent/reports/r6_wide_rank_capacity_analysis.md`),
and integrated into the Agent's methodology-relevant records. **R6-15
remains the canonical controlled comparison; R6-16 does NOT replace it** —
it is a capacity-sensitivity ablation/control (status: `ablation`).

Key finding recorded (use this EXACT qualified phrasing, not an unqualified
"capacity does not explain the gap"): **"the tested increase in RankExt
capacity does not appear to be the dominant limiting factor within the
evaluated range."** All_seen accuracy gains were +0.6 to +1.6pp across the
four RankExt variants; first-step accuracy (no forgetting yet) got WORSE for
all four despite doubled step-1 rank; the ~30-40pp non-KD SimpleAvg gap
closed by under 5%.

Files changed this pass:
- `thesis_agent/scripts/build_knowledge_base.py` — **the single source of
  truth** — gained: one new `add_exp(experiment_id="R6-16", ...)` block
  (right after R6-15); claim `C04` (increasing rank capacity does not solve
  the RankExt bottleneck) and `C10` (SimpleAvg/RankExt not capacity-matched)
  both extended with R6-16 as independent, more-controlled (matched
  target-module, canonical-protocol) confirmation; `CONFLICT-005` (wide
  schedule "more params" mislabeling) extended — the SAME
  `rankext_more_params_than_simple_avg` auto-flag imprecision recurs in
  R6-16's own run_config.json; `CONFLICT-011` (stale RQ1 target-module
  parenthetical) got a note that RankExt's provable rank cap is schedule-
  dependent (80 under canonical, 160 under R6-16), without altering the
  locked RQ1 wording. **Re-ran the script** — regenerated
  `data/experiments.jsonl` (65), `data/claims.jsonl` (17),
  `data/conflicts.jsonl` (11), `data/decisions.jsonl` (15),
  `data/figures_tables.jsonl` (10), `data/supervisor_requirements.json` —
  all now durable (survive a future re-run of the build script).
- `thesis_agent/data/methods.json` (hand-maintained input, NOT a
  build-script output) — `families.rank_extension.rank_schedule_current`
  corrected to distinguish "canonical schedule value" from "currently
  checked-out flag state" (as of commit `e66b949`/HEAD,
  `USE_RANKEXT_RANK_SCHEDULE_WIDE=True` is actually active — the checked-out
  `.py` would currently reproduce R6-16's wide schedule, NOT R6-15's
  canonical one, if re-run as-is); `disabled_or_abandoned_methods.
  wide_rank_schedule.status` corrected from a flat "REVERTED" (stale — true
  only for the OLDER 2026-07-21 5x20-era test) to document BOTH the reverted
  first test and the retained, currently-active R6-16 second test;
  `chapter_3_destinations` gained `capacity_asymmetry_disclosure`.
- `thesis_agent/chapter_3/chapter_3_code_truth.md` §M and §O — added a
  STALENESS CORRECTION block (the evidence pack, built 2026-09-01, predates
  the 2026-09-02 flag flip and incorrectly implied `USE_RANKEXT_RANK_SCHEDULE_
  WIDE=False` is the current state); §N's canonical-experiment-context table
  gained an explicit R6-16 row (capacity-sensitivity control, NOT canonical);
  §O's historical/rejected list now distinguishes the genuinely-reverted
  2026-07-21 5x20 wide test from the currently-retained 2026-09-02 4x25
  R6-16 control (which does NOT belong in a "rejected" list).
- `thesis_agent/reports/canonical_results.md` — R6-16 added to the "DO NOT
  substitute" list and the protocol-specific reference table (category 3,
  ablation), R6-15 untouched as canonical.
- `thesis_agent/sources/R6_catalog.md` — entry #17 (R6-16) and cross-run
  observation #5 both had their "capacity does NOT explain the gap" /
  "does not explain its gap" headings corrected to the qualified
  tested-range phrasing above.
- `thesis_agent/reports/r6_wide_rank_capacity_analysis.md` — tightened §6's
  answer paragraph to lead with the exact qualified sentence.
- Re-ran `scripts/validate_knowledge_base.py` after every data change:
  **0 errors, 5 warnings (all pre-existing/unrelated to R6-16), 17 passes**
  throughout.

**Not done, and deliberately out of scope for this pass:** the `.py`'s
`USE_RANKEXT_RANK_SCHEDULE_WIDE` flag was NOT flipped back to `False`. That
is a training-run/experiment-management decision, not a documentation-sync
task — flip it back manually before any byte-for-byte canonical rerun
(n5.py:1244 already documents how). Chapter 3/4/5 prose was NOT written or
modified — this was a synchronization-only pass, per explicit instruction.

SIMPLEAVG / RANKEXT / KD / FACTORORTH / CALIBRATION / OPEN-RESTRICTED /
METRICS / CAPACITY DISCLOSURE code-truth status: **READY.** Safe to start
writing Chapter 3 from the Agent: **YES** (see full final report in this
session's chat transcript for the itemized checklist).

---

## 2026-09-01 — Chapter 3 Methodology preparation + synchronization pass (COMPLETE)

A full source-audit pass re-verified every methodology mechanism line-by-line
against `vit_lora_cifar100_full5step_n5.py` @ HEAD `8b2877e`. Outputs:
- **New: `thesis_agent/chapter_3/`** — 6-file evidence pack + README
  (methodology_map, code_truth, equation_map, algorithm_map, method_variants,
  open_questions). This is the Chapter 3 evidence pack. `chapter_3.tex` NOT
  written — deliberately.
- `data/conflicts.jsonl` +CONFLICT-010 (n5 notebook is a stale 5×20 July-2026
  snapshot; `.py` is canonical) and +CONFLICT-011 (stale RQ1
  "2 vs 4 target modules" wording — both final families use `["q_proj","v_proj"]`).
- `reports/candidate_research_questions.md` RQ1 scope note corrected (wording
  of the RQ itself untouched — still locked).
- `data/methods.json` — added `notebook_divergence`, `chapter_3_destinations`;
  the two previously-`UNVERIFIED` items (`build_simple_avg_teacher_model`,
  `average_factor_reference_state`) marked VERIFIED with exact behaviour.
- `sources/method_ground_truth.md` §11 UNVERIFIED items resolved; new §12
  (notebook vs executable comparison).
- `scripts/validate_knowledge_base.py` — +Checks 14–17 (target-module
  currency, Chapter 3 pack presence, stale RQ1 wording gone, forward_transfer
  stays REJECTED). Not weakened.
- Validation after: **0 errors, 5 warnings (all pre-existing/benign), 17 passes.**

SIMPLEAVG / RANKEXT / KD / FACTORORTH / CALIBRATION / OPEN-RESTRICTED /
METRICS code-truth status: **READY.** Safe to start writing Chapter 3 from
the Agent: **YES.**

---

## 1. Current completed state

The knowledge base is fully built and validated:
- `sources/` — 7 primary research dossiers (tier 1-2 evidence), complete.
- `data/` — `experiments.jsonl` (64 entries), `decisions.jsonl` (15),
  `claims.jsonl` (16), `conflicts.jsonl` (9, all resolved),
  `figures_tables.jsonl` (10), `methods.json`, `sources_manifest.json`,
  `supervisor_requirements.json` — all generated by
  `scripts/build_knowledge_base.py` (single source of truth; re-run it to
  regenerate `data/*` if a correction is needed — edit the script, not the
  generated files).
- `chapters/` — all 6 chapter evidence-maps complete.
- `reports/` — `canonical_results.md`, `optimized_results.md`,
  `failed_experiments.md`, `open_questions.md`, `evidence_gaps.md`,
  `project_timeline.md`, `thesis_readiness_report.md`,
  `candidate_research_questions.md`, `candidate_contributions.md` — all
  present and complete.
- `scripts/` — `build_knowledge_base.py`, `validate_knowledge_base.py`,
  `query_thesis_agent.py`, all tested working. Last validation run: **0
  errors, 5 warnings (all benign), 13 passes.**
- `README.md` — top-level entry point, complete.

An earlier build-process incident (a background subagent scope-crept and
built a competing parallel version of this knowledge base under a different
ID scheme, causing repeated file-overwrite collisions with the orchestrating
session) was diagnosed and resolved by adopting the subagent's more-complete
version as canonical and regenerating everything from `build_knowledge_base.py`
to a single consistent schema. Full incident record was previously in this
file; superseded by this checkpoint. No further action needed on that
incident.

A full canonicality audit (2026-08-29) was completed and the result is
recorded in section 2 below — no data or file changes resulted from it
directly; two documentation refinements were recommended and are pending
(section 3).

## 2. Canonicality policy (audited and CONFIRMED — do not re-litigate without new evidence)

- **`R6-15`** (`R6/results_4920359_final8_v2_light`, job 4920359, commit
  `8b2877e`/HEAD) = **canonical controlled comparison**. Confirmed correct
  after an explicit audit against the alternative (R6-14): R6-15 is the
  more internally-fair run (matched calibration algorithm AND matched
  combined-method strength across both families), which is the primary
  requirement for a head-to-head ranking table. Use for any "which method
  is best" claim.
- **`R6-13`** (`results_4918131_light`, job 4918131, **74.07%**
  RankExt+FactorOrth+KD, 9 epochs, RankExt-only run) = **optimized RankExt
  flagship**. Confirmed correct — protocol-matched (4x25) but not
  epoch-matched or method-set-matched to R6-15; cite separately, never
  substitute into the canonical table.
- **`R6-14`** (`results_4918535_final8_light`, job 4918535, SimpleAvg
  **78.34%**) = **historical/superseded, NOT canonical and NOT usable as an
  "optimized SimpleAvg" citation either.** It has two real, documented
  internal-fairness defects (mismatched calibration algorithm between
  families; combined methods deliberately run at half strength) that
  disqualify it from both roles, not just from being "not the latest." No
  valid "optimized SimpleAvg flagship" number currently exists anywhere in
  the project (see refinement 1 below).

Full reasoning: this conversation's canonicality-audit turn (not yet copied
into a `reports/` file — see refinement below).

## 3. Two pending documentation refinements (NOT yet applied)

1. **Clarify the optimized-SimpleAvg policy** in `optimized_results.md`:
   state explicitly that no valid "optimized SimpleAvg" number exists
   (unlike RankExt's clean R6-13), and strengthen the existing 78.34%
   warning so it also blocks use as a "best observed" citation, not just as
   a controlled-comparison citation.
2. **Add the 9->7 epoch quasi-ablation finding** to `chapter_5_results.md`
   §5.3 and to `open_questions.md`/`evidence_gaps.md`: R6-14->R6-15 isolates
   epochs-only for RankExt's 3 non-combined methods (calibration and
   combined-strength don't touch them) — result is mixed/small
   (`rank_extension` -2.68pp, `rank_extension_kd_only_T2` -1.18pp,
   `rank_extension_orth_factor_lam_50` +2.37pp), and a second near-clean
   comparison (R6-13 74.07% @9ep vs. R6-15's same method 73.62% @7ep, both
   full-strength) shows a small -0.45pp cost. Neither is confirmatory
   (single seed) but both should be recorded as evidence partially
   informing (not resolving) the currently-TENTATIVE EPOCHS=7 justification.

## 4. Exact files that still need modification (for the two refinements above)

- `thesis_agent/reports/optimized_results.md` (refinement 1)
- `thesis_agent/reports/open_questions.md` (refinement 2)
- `thesis_agent/reports/evidence_gaps.md` (refinement 2)
- `thesis_agent/chapters/chapter_5_results.md` §5.3 (refinement 2)
- Optionally: a new claim entry in `data/claims.jsonl` (via
  `scripts/build_knowledge_base.py`, not hand-edited) capturing the
  epoch-isolation finding as its own TENTATIVE claim with proper
  evidence/counter-evidence framing, if the refinement should be traceable
  as a first-class claim rather than only prose in the reports above.

## 5. Validation that remains to be run

After the above edits are made:
- If `claims.jsonl` gains a new entry via `build_knowledge_base.py`, re-run
  `python thesis_agent/scripts/build_knowledge_base.py` then
  `python thesis_agent/scripts/validate_knowledge_base.py` and confirm still
  0 errors (warning count may change by 1 if claim-status distribution
  shifts — expected, not a failure).
- Spot-check `chapter_5_results.md` and `open_questions.md` render
  correctly and cross-reference the new claim ID (if added) correctly.
- No other validation is currently outstanding — last full run was clean.

## 6. Next task after those edits

**Review candidate Research Questions and Contributions**
(`thesis_agent/reports/candidate_research_questions.md` and
`thesis_agent/reports/candidate_contributions.md`) — these are explicitly
marked "Not finalized... pending supervisor/author review" and have not yet
been reviewed/approved. Do NOT start this review now; it is queued for
after the two documentation refinements above are applied and validated.

## 7. ImageNet-100 generalization study — PREPARED, NOT LAUNCHED (2026-09-08)

A second, independent preparation task (not related to items 1-6 above, which
remain outstanding and unreordered by this) produced a full engineering/
experiment-design pass for a second-dataset generalization study: does the
SimpleAvg-vs-RankExt / RankExt KD-FactorOrth picture reproduce on ImageNet-100
(same 100-class / 4x25 protocol, same CLIP ViT-B/16 backbone) or is it
CIFAR-100-specific? Full detail:
`thesis_agent/reports/imagenet100_generalization_preparation.md`.

**Status: source prepared and statically verified; NO job submitted; NO
training run.** `vit_lora_cifar100_full5step_n5.py` was restored to its
neutral canonical default (seed 42, `DATASET_NAME="cifar100"`, `RUN_NAME_BASE`
parameterized by dataset/label/seed) and extended with `DATASET_REGISTRY` +
the ImageNet-100 class-definition constants + a consolidated
experiment-invariant assertion block for the *planned* imagenet100 path —
SimpleAvg/RankExt/KD/FactorOrth/calibration/metrics code remains untouched.

**UPDATE (2026-09-08, same day): benchmark definition FINALIZED.** Full
ImageNet-1k confirmed available on the UniPD cluster
(`/nfsd/lttm4/datasets/ImageNet-1k_torch`), removing dataset-availability as a
selection criterion. A deeper CIL-literature audit (downloading and diffing
the actual PODNet/DER/DyTox source files, not inferring from the benchmark
name) found:
- PODNet, DER, and DyTox provably share the exact same 100-class ImageNet
  subset and label mapping (DyTox's file is byte-identical to PODNet's; DER's
  README explicitly defers to PODNet's file).
- That subset is `sorted(all 1000 ImageNet-1k WNIDs)[:100]` (n01440764
  through n01855672) — fully recovered and verified (100/100 unique,
  well-formed WNIDs).
- Its overlap with the previously-default CMC/Tian-et-al. list is only 8/100
  (Jaccard 0.0417) — confirmed via literal programmatic comparison to be two
  unrelated benchmarks sharing a name, not two orderings of one benchmark.
- **`IMAGENET100_SYNSETS` in the source was REPLACED** with this
  PODNet/DER/DyTox list (CIL-literature precedent now decisively favors it,
  and reproducibility/access are no longer differentiating). The original CMC
  list is retained, inactive, as `IMAGENET100_SYNSETS_CMC_ALTERNATIVE`.
- A seed-42 deterministic permutation (`IMAGENET100_CLASS_ORDER`) was
  generated for the 4x25 task-order assignment (no canonical published task
  order was confirmed for this benchmark).
- The loader was switched from a Hugging Face Hub mirror to a local
  `imagefolder` load against `IMAGENET_ROOT` (env-var, default the verified
  cluster path), filtered to the 100 selected WNIDs, with an explicit
  label-remap step guarding against the loader's own indices (or the
  original ImageNet-1k 0-999 indices) leaking into the classifier.
- A standalone verification script (`scripts/verify_imagenet100_local.py`)
  was written and sandbox-tested, but **not yet run against the real cluster
  path** (no cluster access from this session) — this is the next concrete
  step, expected to report 0 missing classes.

Full detail: `thesis_agent/reports/imagenet100_generalization_preparation.md`
Section 0, "Final ImageNet-100 Benchmark Selection".

**Remaining open item:** run `scripts/verify_imagenet100_local.py` on the
cluster and confirm 0 missing classes, then perform the real (not static)
smoke test described in the preparation report before considering launch.

---

SAFE CHECKPOINT REACHED: YES
NEXT TASK (two independent, unordered pending items):
(a) Apply the two pending documentation refinements from items 1-6 above
    (optimized-SimpleAvg policy clarification in optimized_results.md;
    9->7 epoch quasi-ablation finding in chapter_5_results.md/open_questions.md/
    evidence_gaps.md), re-validate, then review candidate_research_questions.md
    and candidate_contributions.md; OR
(b) Run `scripts/verify_imagenet100_local.py` on the UniPD cluster against
    `/nfsd/lttm4/datasets/ImageNet-1k_torch` (class-list choice is now
    FINALIZED -- PODNet/DER/DyTox lineage, Section 0 of the preparation
    report; this step just confirms local presence), then the real (not
    static) smoke test, before any ImageNet-100 job submission.
Neither is implicitly prioritized over the other by this file -- follow
whichever the author asks for next.
