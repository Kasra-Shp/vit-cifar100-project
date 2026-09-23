# CIFAR-100 job4971615 normalized Combined full report

STATUS: PENDING — the full Slurm experiment has been prepared but not submitted.

Canonical source SHA256 before/after must remain:

`B6E9C95377F0506726679782E81E25F2A4780D28D393679E07AF1118CEBBB555`

| Variant | All-seen | Restricted | Δ vs canonical |
| --- | ---: | ---: | ---: |
| Canonical Combined | 70.76% | 94.78% | 0.00 pp |
| Normalized Combined | pending | pending | pending |
| Normalized Combined + Calibration | pending | pending | pending |

Method: `rank_extension_normalized_factor_orth_lam50_fullkd_T2_protect30`

Protocol: CIFAR-100, 5×20, canonical class order, seed 42, 9 epochs/task, rank schedule `[16,32,48,64,80]`, canonical preprocessing/optimizer, RankExt head LR 1e-4, full KD T=2 weight 1, Protect30, normalized FactorOrth λ=50.

The offline calibration script will replace the pending values after the run using validation-only fitting and saved logits only.
