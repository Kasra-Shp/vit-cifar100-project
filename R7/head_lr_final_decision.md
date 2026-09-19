# Head-LR Multiplier: Evidence Audit and Final Decision (for `final_8method_5x20_refined.py`)

## Evidence checked

**Repository code and its own historical comments** (`experiments_prepared/final_9method_5x20_performance_recovery.py`,
lines 968–999, verbatim, unchanged by this task):

```
HEAD_LR_MULTIPLIER = 10.0
```
with the dated comment directly above it:

> "REVERT (2026-07-16, analysis_rankext_firststep/report.txt): now FAMILY-CONDITIONAL... rank_extension
> reverts to x1.0 (BASELINE, no multiplier) -- the report flagged head_lr x10 as a plausible AMPLIFIER
> (not sole cause) of the transient step-boundary CE spike that factor-orth enlarges for
> rank_extension, since a 10x classifier LR turns a noisy transient loss spike into a much larger
> one-shot weight change. It was REJECTED as a sufficient cause on its own (applied uniformly in the
> calibfix run to all 4 rank_extension variants, only 2 of which collapsed), but reverting it removes
> one more untested variable while we test the lambda-warmup fix below, and BASELINE (x1.0) is the
> config that actually produced rank_extension's 68.0 historical best. simple_avg keeps x10 (never
> implicated; SimpleAvg+FactorOrth's 75.5% was achieved WITH it)."

And, immediately below:

> "RESTORED (2026-08-25, explicit user correction): simple_avg's head-LR multiplier reverted back to
> x10, matching the R6 reference file byte-for-byte... rank_extension's own head-LR multiplier (x1.0,
> its own proven-safe BASELINE...) is UNTOUCHED by this revert."

```
HEAD_LR_MULTIPLIER_BY_FAMILY = {
    "simple_avg": float(HEAD_LR_MULTIPLIER),   # 10.0
    "rank_extension": 1.0,
}
```

**What this evidence directly establishes:**
- RankExt ×10 was investigated once, historically, and found to be **a plausible amplifier of a
  documented instability** (a transient CE spike at CL-step boundaries, worsened for FactorOrth
  configurations specifically) — explicitly **not** claimed as the sole cause (only 2 of 4 tested
  RankExt variants collapsed under it), but explicitly reverted to ×1 as the safer, "proven-safe
  BASELINE," and that ×1 setting is the one that produced RankExt's own historical best (68.0%).
- SimpleAvg's ×10 has **never been directly implicated in any instability**, and one specific
  SimpleAvg+FactorOrth result (75.5%) was achieved with it — this is evidence of *absence of observed
  harm*, not evidence that ×10 is *necessary* or *beneficial*. **No controlled ×10-vs-×1 ablation for
  SimpleAvg exists anywhere in this repository.**
- The two later, independent capacity-focused experiments (Exp2, R6-16 wide-rank) that varied
  RankExt's rank left head-LR unchanged throughout (RankExt ×1, SimpleAvg ×10) and still found the
  RankExt-vs-SimpleAvg gap barely moved — this shows the gap survives when capacity is controlled, but
  says nothing directly about head-LR's own causal contribution (it was never the varied factor in
  those experiments).
- No dedicated head-LR ablation (for either family) exists anywhere in `R6/`, `R7/`, or the
  `thesis_agent/reports/` corpus searched for this and the immediately preceding audit task.

## External literature check

A short, focused literature check was performed (web search, September 2026) against primary sources
on classifier/head learning-rate effects, old/new classifier bias, and stability-plasticity dynamics
in class-incremental learning (CIL), specifically to check whether external evidence gives any reason
to reverse the shared-×1.0 decision above.

- **Mittal, Galesso & Brox, "Essentials for Class Incremental Learning," CVPR-W 2021**
  ([arXiv:2102.09517](https://arxiv.org/abs/2102.09517)) — directly on point. This paper explicitly
  studies **reducing the learning rate on incremental-step training ("LowLR")** as a mitigation for
  task-recency bias, and finds that a **lower** learning rate on incremental steps reduces classifier
  bias toward the most-recently-learned classes; combined with separated softmax, LowLR gives the
  paper's best class-IL results, and LowLR alone ("Comb+LowLR") already reduces the bias and improves
  performance versus a higher-LR baseline. This is a **directional, not numeric**, finding: it studies
  the overall step learning rate, not a classifier-head-specific multiplier on top of a fixed backbone
  LR, and it does not test anything resembling our ×1 vs. ×10 comparison. But directionally, it points
  the same way as our decision: *lower*, not higher, update magnitude on incremental-step training is
  associated with *less* classifier bias toward recent classes — the opposite of what a ×10
  classifier-head LR would push toward.
- **Zhao et al., "Maintaining Discrimination and Fairness in Class Incremental Learning" (Weight
  Aligning, WA), CVPR 2020** ([arXiv:1911.07053](https://arxiv.org/abs/1911.07053)) — identifies that
  the last fully-connected layer's weights become "highly biased" in CIL (new-class row norms grow
  disproportionately large relative to old-class rows) and corrects this **post-hoc, after training**,
  by rescaling new-class row norms to match old-class row norms. It does not address or test
  classifier-head learning rate at all, but it corroborates the general principle that classifier
  weight *magnitude* is a sensitive, bias-prone axis in CIL — consistent with treating a ×10
  amplification of classifier-weight updates as a plausible risk factor, not a neutral or
  self-evidently beneficial choice.
- **Ahn, Kwak, Lim et al., "SS-IL: Separated Softmax for Incremental Learning," ICCV 2021**
  ([arXiv:2003.13947](https://arxiv.org/abs/2003.13947)) — traces classification score bias to how
  old/new logits are jointly softmax-normalized during training, and fixes it via a separated-softmax
  training objective (an architectural/loss-level change, not a classifier-LR change). Reinforces the
  same general theme (classifier-side score imbalance between old and new classes is a well-documented,
  important CIL failure mode) without saying anything about head-LR multipliers specifically.

**Conclusion of the literature check:** the literature strongly and consistently supports the general
importance of controlling classifier update magnitude / old-vs-new classifier score-and-norm balance
in class-incremental learning, and the one directly-relevant learning-rate-specific finding (LowLR in
Mittal et al.) points toward *lower*, not higher, incremental-step learning rates reducing recency
bias. **None of the surveyed papers establish, test, or imply a universal optimal numeric
classifier-head-LR multiplier**, and none test a PEFT/LoRA classifier-head-multiplier setup resembling
ours. This does not on its own prove ×1.0 is optimal for this codebase — but it gives no reason to
reverse the decision, and if anything mildly reinforces it directionally. **The repository's own
project-specific, dated instability evidence (Section "Evidence checked" above) remains the strongest
and most directly applicable reason for choosing shared ×1.0 here**, exactly as stated before this
literature check was performed; the external literature is corroborating context, not the primary
basis for the decision.

## Conclusion

**A single shared value is preferred over keeping the ×10/×1 asymmetry, and the safer, more
conservative common value is `HEAD_LR_MULTIPLIER = 1.0` for both families.**

## Chosen common multiplier

```
HEAD_LR_MULTIPLIER = 1.0   # applies identically to simple_avg and rank_extension
```

## Why ×10 (for both) was rejected

- For RankExt, ×10 has **documented, dated, incident-specific evidence of being a plausible
  instability amplifier**, specifically in combination with auxiliary losses (FactorOrth) — exactly
  the kind of configuration this new 8-method experiment retains (`rank_extension_factor_orth_lam50`,
  and the FactorOrth+KD combined arm). Applying ×10 to RankExt here would reintroduce a factor the
  repository's own history already flagged and removed.
- For SimpleAvg, there is no controlled evidence ×10 is *necessary*: the cited historical result
  (75.5% with ×10) is a single existence proof, not a comparison against ×1. Absence of documented
  harm is not evidence of benefit.
- ×1 for both is the strictly safer choice with respect to the one specific, documented failure
  mode in this codebase, and it removes an unexplained, un-ablated structural asymmetry between the
  two families being compared — this directly serves the fairness/interpretability goal of the new
  8-method experiment.

## Limitations

- **This is not a dedicated LR ablation.** No new experiment was run to directly compare ×1 vs. ×10
  for either family under the current (9-epoch, protect30-bearing, FactorOrth-bearing) configuration.
  The decision rests entirely on repository-documented historical evidence and a conservative-default
  heuristic ("prefer the setting with no documented failure mode over the one with a documented,
  if non-sufficient, failure mode"), not a new controlled measurement.
- It remains possible that SimpleAvg's absolute accuracy is somewhat lower under ×1 than under ×10 in
  this new experiment — this was not tested, and the new experiment's SimpleAvg numbers should be read
  with that in mind if compared directly against job 4971615's ×10-based SimpleAvg results.
- The instability evidence for RankExt+FactorOrth under ×10 came from a different (older) training
  configuration; whether it still applies unchanged under the current 9-epoch/protect30 setup is
  assumed, not re-verified, by this decision.
