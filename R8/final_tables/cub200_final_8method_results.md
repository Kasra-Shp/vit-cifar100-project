# Final CUB-200 8-method results

| Method | All-seen (%) | Restricted (%) | BWT | Forgetting |
|---|---|---|---|---|
| SimpleAvg | 69.468 | 91.251 | -0.241 | — |
| SimpleAvg + KD | 68.692 | 91.942 | -0.266 | — |
| SimpleAvg + DenseOrth | 68.329 | 91.047 | -0.259 | — |
| SimpleAvg + DenseOrth + KD | 67.242 | 91.735 | -0.288 | — |
| RankExt | 20.262 | 78.120 | -0.891 | 0.890 |
| RankExt KD+Protect | 45.271 | 87.779 | -0.547 | 0.589 |
| RankExt Normalized FactorOrth | 20.815 | 79.581 | -0.885 | 0.884 |
| RankExt Normalized KD+FactorOrth | 63.428 | 88.385 | -0.532 | 0.570 |

Note: the final RankExt Normalized KD+FactorOrth All-seen/Restricted values are the selected task-block-scale result. Its BWT/Forgetting values remain the raw training-trajectory metrics because calibration was applied only at final evaluation.
