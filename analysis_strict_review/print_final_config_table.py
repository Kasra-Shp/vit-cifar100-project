"""
Prints the full final per-method config table for the NEXT instrumented run,
computed from the ACTUAL current values of every relevant constant in
vit_lora_cifar100_full5step_n5.py (parsed directly out of the source file so
there is no risk of a hand-transcribed number drifting from the real code).
No training, no model/dataset loading -- this only replicates the small pure
config-assembly logic (family_target_modules / family_head_lr_multiplier /
family_applies_calibration / add_method's per-method scale application),
copied verbatim in spirit from the script, over constants read from disk.
"""
import re
import json
from pathlib import Path

SRC = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/vit_lora_cifar100_full5step_n5.py").read_text(encoding="utf-8")


_ENV = {}


def const(name, cast=float):
    m = re.search(rf"^{name}\s*=\s*([^\n#]+)", SRC, re.MULTILINE)
    assert m, f"could not find constant {name}"
    val = m.group(1).strip()
    result = cast(eval(val, {}, _ENV))
    _ENV[name] = result
    return result


SEED = const("SEED", int)
LORA_R = const("LORA_R", int)
LORA_ALPHA = const("LORA_ALPHA", int)
LORA_DROPOUT = const("LORA_DROPOUT", float)
LAMBDA_ORTH = const("LAMBDA_ORTH", float)
KD_WEIGHT = const("KD_WEIGHT", float)
LORA_EPOCHS = const("LORA_EPOCHS", int)
RANKEXT_EPOCHS = const("RANKEXT_EPOCHS", int)
LR_LORA = const("LR_LORA", float)
LR_RANKEXT = const("LR_RANKEXT", float)
COMBINED_LOSS_SCALE_ENABLED = const("COMBINED_LOSS_SCALE_ENABLED", bool)
COMBINED_LAMBDA_ORTH_SCALE = const("COMBINED_LAMBDA_ORTH_SCALE", float)
COMBINED_KD_WEIGHT_SCALE = const("COMBINED_KD_WEIGHT_SCALE", float)
COMBINED_ORTH_WARMUP_ENABLED = const("COMBINED_ORTH_WARMUP_ENABLED", bool)
COMBINED_ORTH_WARMUP_EPOCHS = const("COMBINED_ORTH_WARMUP_EPOCHS", float)
RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED = const("RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED", bool)
RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS = const("RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS", float)
USE_RANKEXT_RANK_SCHEDULE_WIDE = const("USE_RANKEXT_RANK_SCHEDULE_WIDE", bool)

def multiline_dict(name, extra_env=None):
    m = re.search(rf"^{name}\s*=\s*(\{{.*?\n\}})", SRC, re.MULTILINE | re.DOTALL)
    assert m, f"could not find dict constant {name}"
    env = dict(extra_env or {})
    return eval(m.group(1), {}, env)


m = re.search(r'^TARGET_MODULES\s*=\s*(\[[^\n]+\])', SRC, re.MULTILINE)
TARGET_MODULES = eval(m.group(1))
TARGET_MODULES_BY_FAMILY = multiline_dict("TARGET_MODULES_BY_FAMILY", {"TARGET_MODULES": TARGET_MODULES, "list": list})
HEAD_LR_MULTIPLIER_BY_FAMILY = multiline_dict("HEAD_LR_MULTIPLIER_BY_FAMILY", {"HEAD_LR_MULTIPLIER": const("HEAD_LR_MULTIPLIER", float), "float": float})
CALIBRATION_ENABLED_FAMILIES = multiline_dict("CALIBRATION_ENABLED_FAMILIES")
m = re.search(r"^RANKEXT_RANK_SCHEDULE\s*=\s*(\[[^\n]+\])", SRC, re.MULTILINE)
RANKEXT_RANK_SCHEDULE = eval(m.group(1))
m = re.search(r"^RANKEXT_RANK_SCHEDULE_WIDE\s*=\s*(\[[^\n]+\])", SRC, re.MULTILINE)
RANKEXT_RANK_SCHEDULE_WIDE = eval(m.group(1))

active_rank_schedule = RANKEXT_RANK_SCHEDULE_WIDE if USE_RANKEXT_RANK_SCHEDULE_WIDE else RANKEXT_RANK_SCHEDULE

METHODS = [
    ("simple_avg", "simple_avg", False, False),
    ("simple_avg_kd_T2", "simple_avg", True, False),
    ("simple_avg_factor_orth", "simple_avg", False, True),
    ("simple_avg_factor_orth_kd_T2", "simple_avg", True, True),
    ("rank_extension", "rank_extension", False, False),
    ("rank_extension_kd_only_T2", "rank_extension", True, False),
    ("rank_extension_orth_factor_lam_50", "rank_extension", False, True),
    ("rank_extension_orth_factor_lam_50_kd_T2", "rank_extension", True, True),
]

rows = []
for method, family, uses_kd, uses_orth in METHODS:
    is_combined_simple = method == "simple_avg_factor_orth_kd_T2"
    lambda_scale = COMBINED_LAMBDA_ORTH_SCALE if (is_combined_simple and COMBINED_LOSS_SCALE_ENABLED) else 1.0
    kd_scale = COMBINED_KD_WEIGHT_SCALE if (is_combined_simple and COMBINED_LOSS_SCALE_ENABLED) else 1.0
    row = {
        "method": method,
        "family": family,
        "target_modules": ",".join(TARGET_MODULES_BY_FAMILY[family]),
        "head_lr_multiplier": HEAD_LR_MULTIPLIER_BY_FAMILY[family],
        "apply_calibration": CALIBRATION_ENABLED_FAMILIES[family],
        "num_epochs": RANKEXT_EPOCHS if family == "rank_extension" else LORA_EPOCHS,
        "learning_rate": LR_RANKEXT if family == "rank_extension" else LR_LORA,
        "lora_rank": (active_rank_schedule[-1] if family == "rank_extension" else LORA_R),
        "rank_schedule": ("->".join(map(str, active_rank_schedule)) if family == "rank_extension" else f"fixed:{LORA_R}"),
        "uses_kd": uses_kd,
        "kd_temperature": 2.0 if uses_kd else 0.0,
        "kd_weight": (KD_WEIGHT * kd_scale) if uses_kd else 0.0,
        "uses_factor_orth": uses_orth,
        "lambda_orth": (LAMBDA_ORTH * lambda_scale) if uses_orth else 0.0,
        "lambda_orth_scale": lambda_scale,
        "kd_weight_scale": kd_scale,
        "combined_orth_warmup_enabled": COMBINED_ORTH_WARMUP_ENABLED if is_combined_simple else False,
        "combined_orth_warmup_epochs": COMBINED_ORTH_WARMUP_EPOCHS if is_combined_simple else None,
        "rankext_orth_lambda_warmup_enabled": (RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED if (family == "rank_extension" and uses_orth) else False),
        "rankext_orth_lambda_warmup_epochs": (RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS if (family == "rank_extension" and uses_orth) else None),
        "rankext_rank_schedule_wide_enabled": USE_RANKEXT_RANK_SCHEDULE_WIDE,
        "seed": SEED,
    }
    rows.append(row)

import pandas as pd
df = pd.DataFrame(rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
print(df.to_string(index=False))
df.to_csv(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/analysis_strict_review/final_per_method_config_table_for_next_run.csv", index=False)
