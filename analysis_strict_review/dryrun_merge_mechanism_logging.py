"""
Dry-run of the B1 merge-mechanism logging function (log_merge_mechanism, added
to vit_lora_cifar100_full5step_n5.py) against synthetic small matrices, since
no training is run this session. Exercises:
  (a) a "healthy" case: 5 near-random, near-orthogonal task deltas of similar
      norm -> expect merged_norm_over_mean_individual_norm close to 1/sqrt(5)
      (textbook variance-reduction from averaging independent random
      directions), cos(dW1, dWt>1) near 0, cos(dW1, merged) moderate.
  (b) a "destructive dilution" case, engineered to mimic the hypothesis under
      test: tasks 2-5 deliberately built with a component that cancels task
      1's direction -> expect cos(dW1, merged) strongly reduced vs cos(dW1,
      dW1)=1, and merged_norm_over_mean_individual_norm well below 1.
This validates the logging function's arithmetic and CSV output shape only --
it says nothing about which case actually occurs in training (that requires
the real instrumented run and tables/merge_mechanism_by_method_step.csv).
"""
import sys
import numpy as np
import torch
import pandas as pd
import os

sys.path.insert(0, r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/analysis_strict_review")

# ---- reimplement log_merge_mechanism verbatim (no import of the full script) ----
_MERGE_MECHANISM_CSV_INITIALIZED = set()


def log_merge_mechanism(method_name, step_states, merged_delta, csv_path):
    if len(step_states) == 0:
        return None
    keys = sorted(step_states[0]["deltas"].keys())
    n_tasks = len(step_states)
    rows = []
    for key in keys:
        task_deltas = [step_states[t]["deltas"].get(key) for t in range(n_tasks)]
        task_deltas = [d.float() if d is not None else None for d in task_deltas]
        dW1 = task_deltas[0]
        norm1 = float(torch.linalg.norm(dW1)) if dW1 is not None else float("nan")
        individual_norms = [float(torch.linalg.norm(d)) for d in task_deltas if d is not None]
        mean_individual_norm = float(np.mean(individual_norms)) if individual_norms else float("nan")
        for t, dWt in enumerate(task_deltas):
            if dWt is None:
                continue
            norm_t = float(torch.linalg.norm(dWt))
            if dW1 is not None and norm1 > 0 and norm_t > 0:
                cos_1t = 1.0 if t == 0 else float(torch.dot(dW1.reshape(-1), dWt.reshape(-1)) / (norm1 * norm_t))
            else:
                cos_1t = float("nan")
            rows.append({"method": method_name, "target_module": key, "task_step": t + 1, "phase": "pre_merge",
                         "dW_norm": norm_t, "dW_norm_over_dW1_norm": (norm_t / norm1) if norm1 > 0 else float("nan"),
                         "cos_dW1_dWt": cos_1t, "n_tasks_in_merge": n_tasks})
        merged = merged_delta.get(key)
        if merged is not None:
            merged_norm = float(torch.linalg.norm(merged))
            cos_1_merged = (float(torch.dot(dW1.reshape(-1), merged.reshape(-1)) / (norm1 * merged_norm))
                            if (dW1 is not None and norm1 > 0 and merged_norm > 0) else float("nan"))
            rows.append({"method": method_name, "target_module": key, "task_step": "MERGED", "phase": "post_merge",
                         "dW_norm": merged_norm, "dW_norm_over_dW1_norm": (merged_norm / norm1) if norm1 > 0 else float("nan"),
                         "cos_dW1_dWt": cos_1_merged, "n_tasks_in_merge": n_tasks,
                         "merged_norm_over_mean_individual_norm": (merged_norm / mean_individual_norm if mean_individual_norm > 0 else float("nan"))})
    df = pd.DataFrame(rows)
    write_header = csv_path not in _MERGE_MECHANISM_CSV_INITIALIZED
    if write_header and os.path.exists(csv_path):
        os.remove(csv_path)
    _MERGE_MECHANISM_CSV_INITIALIZED.add(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    return df


def simple_average_deltas(step_states):
    keys = sorted(step_states[0]["deltas"].keys())
    merged = {}
    for key in keys:
        vals = [s["deltas"][key].float() for s in step_states if key in s["deltas"]]
        merged[key] = torch.stack(vals, dim=0).mean(dim=0)
    return merged


torch.manual_seed(0)
np.random.seed(0)
OUT_CSV = r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/analysis_strict_review/dryrun_merge_mechanism.csv"
if os.path.exists(OUT_CSV):
    os.remove(OUT_CSV)

# --- case (a): healthy, near-orthogonal random deltas, similar norm ---
n_tasks, rows_, cols_ = 5, 8, 6
healthy_states = []
for t in range(n_tasks):
    d = torch.randn(rows_, cols_) * 0.1
    healthy_states.append({"deltas": {"q_proj": d}})
merged_healthy = simple_average_deltas(healthy_states)
log_merge_mechanism("dryrun_healthy", healthy_states, merged_healthy, OUT_CSV)

# --- case (b): engineered destructive dilution -- tasks 2-5 each contain
# task 1's own delta PLUS an equal-magnitude, exactly-opposing component from
# a shared "conflict" direction, so their mean partially cancels dW1 ---
dW1 = torch.randn(rows_, cols_) * 0.1
conflict_dir = torch.randn(rows_, cols_) * 0.1
dilution_states = [{"deltas": {"q_proj": dW1.clone()}}]
for t in range(1, n_tasks):
    dilution_states.append({"deltas": {"q_proj": -dW1 * 0.9 + conflict_dir * 0.05}})
merged_dilution = simple_average_deltas(dilution_states)
log_merge_mechanism("dryrun_destructive_dilution", dilution_states, merged_dilution, OUT_CSV)

result = pd.read_csv(OUT_CSV)
pd.set_option("display.width", 200)
print(result.to_string(index=False))

merged_rows = result[result.task_step == "MERGED"]
print("\nSanity checks:")
print("healthy case merged_norm_over_mean_individual_norm ~ 1/sqrt(5) =", 1/np.sqrt(5), "-> logged:",
      merged_rows[merged_rows.method == "dryrun_healthy"]["merged_norm_over_mean_individual_norm"].iloc[0])
print("destructive-dilution case cos(dW1, merged) (should be strongly reduced, near/below 0):",
      merged_rows[merged_rows.method == "dryrun_destructive_dilution"]["cos_dW1_dWt"].iloc[0])
print("destructive-dilution case merged_norm_over_mean_individual_norm (should be << 1):",
      merged_rows[merged_rows.method == "dryrun_destructive_dilution"]["merged_norm_over_mean_individual_norm"].iloc[0])
