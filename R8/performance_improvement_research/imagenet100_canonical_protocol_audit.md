# ImageNet-100 canonical protocol audit

Source of truth: `experiments_prepared/final_9method_5x20_performance_recovery.py`, canonical CIFAR-100 job4971615 methodology. The ImageNet production script is a dedicated copy/port and does not edit the CIFAR source.

| Component | Canonical CIFAR job4971615 | ImageNet-100 | Match |
|---|---|---|---|
| Backbone | `openai/clip-vit-base-patch16` | Same checkpoint, `local_files_only=True` on cluster | PASS |
| Methods | Exactly 8 thesis methods; T4 excluded | Same 4 SimpleAvg + 4 RankExt IDs/display labels | PASS |
| Task count | 5 | 5 | PASS |
| Classes/task | 20 | 20 | PASS |
| Seed | 42 | 42 | PASS |
| Epochs/task | 9 | 9 | PASS |
| SimpleAvg rank | 80 | 80 | PASS |
| SimpleAvg alpha/scaling | 160 / 2 | 160 / 2 | PASS |
| LoRA targets | q/v | q/v | PASS |
| SimpleAvg LoRA LR | `5e-5` | `5e-5` | PASS |
| SimpleAvg head multiplier | ×10 | ×10 | PASS |
| RankExt head multiplier | ×1 | ×1 | PASS |
| Optimizer | AdamW | AdamW | PASS |
| Weight decay | `0.05` | `0.05` | PASS |
| SimpleAvg KD | canonical T2 old-seen KD configuration | T2 old-seen KD configuration | PASS |
| RankExt KD | full KD, T2, weight 1 | full 100-way KD, T2, weight 1 | PASS |
| Protect | RankExt KD arms Protect30 | RankExt KD arms Protect30 | PASS |
| DenseOrth | lambda 20, SimpleAvg arms | lambda 20, SimpleAvg arms | PASS |
| FactorOrth | lambda 50, RankExt arms | historical/non-normalized lambda 50, RankExt arms | PASS |
| RankExt schedule | `[16, 32, 48, 64, 80]` | `[16, 32, 48, 64, 80]` | PASS |
| RankExt LR | `1e-4` | `1e-4` | PASS |
| Replay | none | none | PASS |
| Classifier | fresh independent LoRA + dense-delta arithmetic average and row stitching for SimpleAvg; persistent GrowingRank classifier for RankExt | same | PASS |
| Evaluation | open all-seen, restricted task subsets, first/later task aggregates, BWT/forgetting/transfer diagnostics | same definitions on ImageNet-100 | PASS |
| Calibration policy | post-merge classifier-row calibration is an evaluation-stage operation | retained as a post-merge operation only; never part of the training loss; raw artifacts are exported for later CPU-only calibration | PASS |
| Data root | canonical CIFAR source owns its dataset path | required external `IMAGENET100_ROOT`; no network fallback | DATASET DIFFERENCE |
| Class identities | CIFAR-100 native labels | prior repository ImageNet-100 WNIDs, exact list saved in package metadata | DATASET DIFFERENCE |
| Image preprocessing | CIFAR-size images | CLIP-compatible RGB, shortest-edge resize, train random crop/flip, deterministic evaluation resize + center crop, CLIP mean/std | UNAVOIDABLE DATASET DIFFERENCE |
| Source splits | CIFAR train/test with canonical validation behavior | ImageNet train plus official held-out split deterministically separated into calibration/test | UNAVOIDABLE DATASET DIFFERENCE |

No normalized FactorOrth variant is used. No task-scale calibration is used in training. The only dataset-specific changes are local file loading, class identity/split metadata, and variable-resolution image preprocessing.
