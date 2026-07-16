import pandas as pd
from pathlib import Path

ROOT = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project")
RUNS = {
    "OLD_20260715": ROOT / "R3/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260715_014445",
    "CALIBFIX_20260716": ROOT / "R3/results_calibfix_20260716_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_133338",
    "NEW_REVERT_20260716": ROOT / "R3/results_revert_20260716_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_193538",
}

OUT = ROOT / "analysis_revert_run"

frames = {}
for tag, path in RUNS.items():
    df = pd.read_csv(path / "tables" / "final_metrics_all_methods.csv")
    df["run"] = tag
    frames[tag] = df

all_df = pd.concat(frames.values(), ignore_index=True)
all_df.to_csv(OUT / "raw_final_metrics_all_runs_long.csv", index=False)

# Wide comparison: one row per method, columns per run per metric
metrics = ["first_step_accuracy", "later_steps_accuracy", "all_seen_accuracy", "forgetting_metric", "backward_transfer"]
methods = sorted(all_df["method"].unique())

rows = []
for m in methods:
    row = {"method": m}
    disp = all_df[all_df.method == m]["display_method_name"].iloc[0] if (all_df.method == m).any() else m
    row["display_method_name"] = disp
    for tag in RUNS:
        sub = frames[tag][frames[tag].method == m]
        for met in metrics:
            row[f"{met}__{tag}"] = sub[met].iloc[0] if len(sub) > 0 and met in sub else float("nan")
    rows.append(row)

wide = pd.DataFrame(rows)

# deltas
for met in metrics:
    wide[f"{met}__delta_CALIBFIX_minus_OLD"] = wide[f"{met}__CALIBFIX_20260716"] - wide[f"{met}__OLD_20260715"]
    wide[f"{met}__delta_NEW_minus_OLD"] = wide[f"{met}__NEW_REVERT_20260716"] - wide[f"{met}__OLD_20260715"]
    wide[f"{met}__delta_NEW_minus_CALIBFIX"] = wide[f"{met}__NEW_REVERT_20260716"] - wide[f"{met}__CALIBFIX_20260716"]

wide.to_csv(OUT / "comparison_all_methods_3runs.csv", index=False)
print(wide[["method"] + [f"first_step_accuracy__{t}" for t in RUNS] + [f"all_seen_accuracy__{t}" for t in RUNS]].to_string())
