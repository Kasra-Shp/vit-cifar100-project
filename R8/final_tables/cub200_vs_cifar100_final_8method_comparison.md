# Final 8-method CIFAR/CUB comparison

| Method family/variant | CIFAR canonical method | CIFAR all-seen (%) | CIFAR restricted (%) | CUB all-seen (%) | CUB restricted (%) | Provenance note |
|---|---|---|---|---|---|---|
| SimpleAvg | SimpleAvg | 75.16 | 92.28 | 69.468 | 91.251 | Canonical CIFAR job4971615. |
| SimpleAvg + KD | SimpleAvg + KD | 75.21 | 92.86 | 68.692 | 91.942 | Canonical CIFAR job4971615. |
| SimpleAvg + DenseOrth | SimpleAvg + FactorOrth | 74.02 | 91.97 | 68.329 | 91.047 | Canonical CIFAR FactorOrth variant; CUB display name is DenseOrth. |
| SimpleAvg + DenseOrth + KD | SimpleAvg + FactorOrth + KD | 73.22 | 92.96 | 67.242 | 91.735 | Canonical CIFAR FactorOrth+KD variant; CUB display name is DenseOrth+KD. |
| RankExt | RankExt | 34.99 | 91.03 | 20.262 | 78.120 | Canonical CIFAR job4971615. |
| RankExt KD+Protect | RankExt + KD + Protect | 65.01 | 94.10 | 45.271 | 87.779 | Canonical CIFAR job4971615. |
| RankExt Normalized FactorOrth | RankExt + FactorOrth | 41.00 | 93.03 | 20.815 | 79.581 | Canonical CIFAR FactorOrth; CUB uses normalized FactorOrth. |
| RankExt Normalized KD+FactorOrth | RankExt + FactorOrth + KD + Protect | 70.76 | 94.78 | 63.428 | 88.385 | CIFAR remains canonical Combined; CUB is normalized KD+FactorOrth with task-block scale calibration. |
