"""
Standalone reproduction of the per-method config resolution logic in
vit_lora_cifar100_full5step_n5.py (add_method() / family_* helpers), using the
SAME constant values as the edited script, for the NEXT run. Does not import
torch/transformers/the dataset -- this script only needs the flag values,
which are copied here 1:1 from the edited .py file (see CHANGES.md for the
exact line references). No training, no GPU. Regenerate
final_per_method_config_table.csv with:
    python final_per_method_config_table.py
"""
import csv

NUM_STEPS = 5
LORA_R = 80
LORA_ALPHA = 160.0
LAMBDA_ORTH = 50.0
KD_WEIGHT = 1.0
KD_TEMPERATURES = [2.0]

RANKEXT_RANK_SCHEDULE = [16, 32, 48, 64, 80]           # default (narrow)
RANKEXT_ALPHA_PER_RANK = 2.0
USE_RANKEXT_RANK_SCHEDULE_WIDE = False                  # TASK C: reverted this session

TARGET_MODULES_BY_FAMILY = {
    "simple_avg": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "rank_extension": ["q_proj", "v_proj"],
}

HEAD_LR_MULTIPLIER_BY_FAMILY = {
    "rank_extension": 1.0,
    "simple_avg": 10.0,
}

RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED = True          # TASK B: the chosen fix, enabled this session

CALIBRATION_ENABLED_FAMILIES = {
    "simple_avg": True,
    "rank_extension": bool(RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED),
}
CALIBRATION_MODE_BY_FAMILY = {
    "simple_avg": "global",
    "rank_extension": "regime_grouped" if RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED else "off",
}

COMBINED_LOSS_SCALE_ENABLED = True
COMBINED_LAMBDA_ORTH_SCALE = 0.3
COMBINED_KD_WEIGHT_SCALE = 0.5


def active_rankext_rank_schedule():
    return list(RANKEXT_RANK_SCHEDULE)  # WIDE is off


def active_rankext_lora_alpha():
    return float(RANKEXT_ALPHA_PER_RANK) * float(active_rankext_rank_schedule()[-1])


def family_target_modules(family):
    return list(TARGET_MODULES_BY_FAMILY.get(family, TARGET_MODULES_BY_FAMILY["simple_avg"]))


def family_head_lr_multiplier(family):
    return HEAD_LR_MULTIPLIER_BY_FAMILY.get(family, 10.0)


def family_applies_calibration(family):
    return bool(CALIBRATION_ENABLED_FAMILIES.get(family, False))


def family_calibration_mode(family):
    return CALIBRATION_MODE_BY_FAMILY.get(family, "global")


def kd_temperature_tag(t):
    t = float(t)
    return f"T{int(t)}" if t.is_integer() else "T" + str(t).replace(".", "p")


rows = []


def add_method(method_name, family, uses_kd=False, kd_temperature=0.0,
                uses_factor_orth=False, lambda_orth_scale=1.0, kd_weight_scale=1.0):
    apply_cal = family_applies_calibration(family)
    rows.append({
        "method": method_name,
        "family": family,
        "uses_kd": uses_kd,
        "kd_temperature": kd_temperature if uses_kd else 0.0,
        "kd_weight": (KD_WEIGHT if uses_kd else 0.0) * kd_weight_scale,
        "uses_factor_orth": uses_factor_orth,
        "lambda_orth": (LAMBDA_ORTH if uses_factor_orth else 0.0) * lambda_orth_scale,
        "rank_schedule": (f"fixed:{LORA_R}") if family == "simple_avg" else "->".join(str(v) for v in active_rankext_rank_schedule()),
        "lora_alpha": LORA_ALPHA if family == "simple_avg" else active_rankext_lora_alpha(),
        "target_modules": ", ".join(family_target_modules(family)),
        "head_lr_multiplier": family_head_lr_multiplier(family),
        "apply_calibration": apply_cal,
        "calibration_mode": family_calibration_mode(family) if apply_cal else "off",
    })


add_method("simple_avg", "simple_avg")
for t in KD_TEMPERATURES:
    add_method(f"simple_avg_kd_{kd_temperature_tag(t)}", "simple_avg", uses_kd=True, kd_temperature=t)
add_method("simple_avg_factor_orth", "simple_avg", uses_factor_orth=True)
_cs = COMBINED_LAMBDA_ORTH_SCALE if COMBINED_LOSS_SCALE_ENABLED else 1.0
_ck = COMBINED_KD_WEIGHT_SCALE if COMBINED_LOSS_SCALE_ENABLED else 1.0
for t in KD_TEMPERATURES:
    add_method(f"simple_avg_factor_orth_kd_{kd_temperature_tag(t)}", "simple_avg", uses_kd=True, kd_temperature=t,
               uses_factor_orth=True, lambda_orth_scale=_cs, kd_weight_scale=_ck)

add_method("rank_extension", "rank_extension")
for t in KD_TEMPERATURES:
    add_method(f"rank_extension_kd_only_{kd_temperature_tag(t)}", "rank_extension", uses_kd=True, kd_temperature=t)
add_method("rank_extension_orth_factor_lam_50", "rank_extension", uses_factor_orth=True)
for t in KD_TEMPERATURES:
    add_method(f"rank_extension_orth_factor_lam_50_kd_{kd_temperature_tag(t)}", "rank_extension", uses_kd=True,
               kd_temperature=t, uses_factor_orth=True)

if __name__ == "__main__":
    fieldnames = list(rows[0].keys())
    with open("final_per_method_config_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in fieldnames}
    header = " | ".join(k.ljust(widths[k]) for k in fieldnames)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[k]).ljust(widths[k]) for k in fieldnames))
