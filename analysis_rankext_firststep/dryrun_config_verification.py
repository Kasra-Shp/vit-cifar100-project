"""
Dry-run verification for the Objective 1 / Objective 2 changes to
vit_lora_cifar100_full5step_n5.py. Executes ONLY the source lines of the
script up through build_active_method_configs() + its assert block (i.e. the
actual config-construction logic, not a re-implementation of it) in an
isolated namespace, stopping well before load_dataset("cifar100") / any
model download or GPU work. No training happens.

Usage: python analysis_rankext_firststep/dryrun_config_verification.py
"""
import ast
import sys

SCRIPT_PATH = "vit_lora_cifar100_full5step_n5.py"
# Cut the source right before this line (exclusive) -- the first statement
# after the config section that isn't needed for config verification and
# would otherwise pull in dataset/model code.
CUTOFF_MARKER = "kd_method_temperature_map = {}"


def load_config_namespace(overrides=None):
    """Executes the script's import + config-construction section (source
    lines only, unmodified) in a fresh namespace. `overrides` is a dict of
    global values to inject BEFORE execution (e.g. to flip a flag like
    USE_RANKEXT_RANK_SCHEDULE_WIDE) by textually patching the corresponding
    assignment line -- done via AST so we don't hand-copy the config logic
    itself, only toggle the same flags a real run would toggle by editing
    the file."""
    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    cutoff_idx = source.index(CUTOFF_MARKER)
    prefix_source = source[:cutoff_idx]

    if overrides:
        tree = ast.parse(prefix_source)
        lines = prefix_source.splitlines(keepends=True)
        # Walk top-level assignments and rewrite the RHS text for any name
        # present in `overrides`, operating on line spans via lineno/end_lineno.
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in overrides:
                    start, end = node.lineno - 1, node.end_lineno
                    new_line = f"{name} = {overrides[name]!r}\n"
                    lines[start:end] = [new_line]
        prefix_source = "".join(lines)

    namespace = {"__name__": "__config_dryrun__"}
    exec(compile(prefix_source, SCRIPT_PATH, "exec"), namespace)
    return namespace


def print_config_table(namespace, title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    cfgs = namespace["ACTIVE_METHOD_CONFIGS"]
    header = (
        f"{'method':<42}{'family':<16}{'target_modules':<22}{'head_lr':>8}"
        f"{'calib':>7}{'lambda_orth':>13}{'kd_weight':>11}{'rank_schedule':>22}"
    )
    print(header)
    print("-" * len(header))
    for cfg in cfgs:
        print(
            f"{cfg['method']:<42}{cfg['family']:<16}{cfg['target_modules']:<22}"
            f"{cfg['head_lr_multiplier']:>8.1f}{str(cfg['apply_calibration']):>7}"
            f"{cfg['lambda_orth']:>13.2f}{cfg['kd_weight']:>11.2f}{cfg['rank_schedule']:>22}"
        )


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    return condition


if __name__ == "__main__":
    all_pass = True

    # ---- 1. Default config (both objectives' default flag states) ----
    ns = load_config_namespace()
    print_config_table(ns, "DEFAULT CONFIG (COMBINED_LOSS_SCALE_ENABLED=True, warmups off, narrow rank schedule)")

    cfg_map = {c["method"]: c for c in ns["ACTIVE_METHOD_CONFIGS"]}

    print("\n--- Objective 1a: family-conditional target_modules / head_lr checks ---")
    all_pass &= check(cfg_map["rank_extension"]["target_modules"] == "q_proj, v_proj",
                       "rank_extension target_modules reverted to q_proj, v_proj")
    all_pass &= check(cfg_map["rank_extension"]["head_lr_multiplier"] == 1.0,
                       "rank_extension head_lr_multiplier reverted to 1.0")
    all_pass &= check(cfg_map["rank_extension_orth_factor_lam_50_kd_T2"]["target_modules"] == "q_proj, v_proj",
                       "rank_extension_orth_factor_lam_50_kd_T2 target_modules reverted")
    all_pass &= check(cfg_map["simple_avg"]["target_modules"] == "q_proj, k_proj, v_proj, out_proj",
                       "simple_avg keeps 4-module target_modules")
    all_pass &= check(cfg_map["simple_avg"]["head_lr_multiplier"] == 10.0,
                       "simple_avg keeps head_lr_multiplier=10.0")
    all_pass &= check(cfg_map["rank_extension"]["apply_calibration"] is False,
                       "rank_extension calibration stays OFF (Objective 1c)")
    all_pass &= check(cfg_map["simple_avg_factor_orth"]["apply_calibration"] is True,
                       "simple_avg calibration untouched (stays ON)")

    print("\n--- Objective 1b: rank schedule + warmup flag checks ---")
    all_pass &= check(ns["active_rankext_rank_schedule"]() == [16, 32, 48, 64, 80],
                       "active_rankext_rank_schedule() is the default (narrow) schedule when flag is off")
    all_pass &= check(cfg_map["rank_extension"]["rank_schedule"] == "16->32->48->64->80",
                       "rank_extension config rank_schedule matches default schedule")
    m1 = ns["orth_lambda_warmup_multiplier"](0.5, 1.0, True)
    m2 = ns["orth_lambda_warmup_multiplier"](2.0, 1.0, True)
    m3 = ns["orth_lambda_warmup_multiplier"](0.5, 1.0, False)
    all_pass &= check(abs(m1 - 0.5) < 1e-9, f"warmup multiplier at epoch=0.5/warmup=1.0 is 0.5 (got {m1})")
    all_pass &= check(abs(m2 - 1.0) < 1e-9, f"warmup multiplier at epoch=2.0/warmup=1.0 clamps to 1.0 (got {m2})")
    all_pass &= check(abs(m3 - 1.0) < 1e-9, f"warmup multiplier is a no-op (1.0) when enabled=False (got {m3})")

    print("\n--- Objective 2: combined loss-scaling checks ---")
    combined = cfg_map["simple_avg_factor_orth_kd_T2"]
    sibling_orth = cfg_map["simple_avg_factor_orth"]
    sibling_kd = cfg_map["simple_avg_kd_T2"]
    all_pass &= check(combined["lambda_orth"] == 25.0, f"combined method lambda_orth scaled to 25.0 (got {combined['lambda_orth']})")
    all_pass &= check(combined["kd_weight"] == 0.5, f"combined method kd_weight scaled to 0.5 (got {combined['kd_weight']})")
    all_pass &= check(sibling_orth["lambda_orth"] == 50.0, "sibling simple_avg_factor_orth KEEPS full-strength lambda_orth=50.0")
    all_pass &= check(sibling_kd["kd_weight"] == 1.0, "sibling simple_avg_kd_T2 KEEPS full-strength kd_weight=1.0")
    all_pass &= check(cfg_map["rank_extension_orth_factor_lam_50_kd_T2"]["lambda_orth"] == 50.0,
                       "rank_extension's KD+orth combo is NOT touched by Objective 2 scaling (pure revert only)")
    all_pass &= check(cfg_map["rank_extension_orth_factor_lam_50_kd_T2"]["kd_weight"] == 1.0,
                       "rank_extension's KD+orth combo kd_weight untouched (1.0)")

    # ---- 2. Flag-toggled configs (prove the flags actually change behavior) ----
    ns_wide = load_config_namespace(overrides={"USE_RANKEXT_RANK_SCHEDULE_WIDE": True})
    cfg_wide = {c["method"]: c for c in ns_wide["ACTIVE_METHOD_CONFIGS"]}
    print("\n--- Flag toggle: USE_RANKEXT_RANK_SCHEDULE_WIDE=True ---")
    all_pass &= check(ns_wide["active_rankext_rank_schedule"]() == [32, 64, 96, 128, 160],
                       "wide schedule resolves to [32,64,96,128,160] when flag is on")
    all_pass &= check(cfg_wide["rank_extension"]["rank_schedule"] == "32->64->96->128->160",
                       "rank_extension config reflects wide schedule when flag is on")
    all_pass &= check(cfg_map["rank_extension"]["rank_schedule"] == "16->32->48->64->80",
                       "default-namespace config is UNCHANGED by the wide-schedule override (no cross-contamination)")

    ns_noscale = load_config_namespace(overrides={"COMBINED_LOSS_SCALE_ENABLED": False})
    cfg_noscale = {c["method"]: c for c in ns_noscale["ACTIVE_METHOD_CONFIGS"]}
    print("\n--- Flag toggle: COMBINED_LOSS_SCALE_ENABLED=False (restores naive full-strength sum) ---")
    all_pass &= check(cfg_noscale["simple_avg_factor_orth_kd_T2"]["lambda_orth"] == 50.0,
                       "combined method reverts to full-strength lambda_orth=50.0 when scaling flag is off")
    all_pass &= check(cfg_noscale["simple_avg_factor_orth_kd_T2"]["kd_weight"] == 1.0,
                       "combined method reverts to full-strength kd_weight=1.0 when scaling flag is off")

    print_config_table(ns_wide, "WIDE RANK SCHEDULE CONFIG (USE_RANKEXT_RANK_SCHEDULE_WIDE=True)")

    # ---- 3. Cross-check against the actual calibfix run's recorded hyperparameters ----
    print(f"\n{'=' * 100}\nCROSS-CHECK vs actual calibfix run (results_calibfix_20260716_light) recorded hyperparameters\n{'=' * 100}")
    import json as _json
    calibfix_path = (
        "R3/results_calibfix_20260716_light/"
        "clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_133338/"
        "configs/hyperparameters_by_method.json"
    )
    with open(calibfix_path) as f:
        calibfix_hparams = {row["method"]: row for row in _json.load(f)}
    print(f"{'method':<42}{'calibfix target_mod':>22}{'NEW target_mod':>16}{'calibfix head_lr':>18}{'NEW head_lr':>12}")
    for method in ["rank_extension", "rank_extension_orth_factor_lam_50_kd_T2", "simple_avg", "simple_avg_factor_orth_kd_T2"]:
        old = calibfix_hparams[method]
        new = cfg_map[method]
        print(
            f"{method:<42}{old['target_modules']:>22}{new['target_modules']:>16}"
            f"{old['head_lr_multiplier']:>18.1f}{new['head_lr_multiplier']:>12.1f}"
        )
    all_pass &= check(
        calibfix_hparams["rank_extension"]["target_modules"] == "q_proj, k_proj, v_proj, out_proj",
        "calibfix run (pre-revert) had rank_extension on 4 modules, confirming the revert is a real change",
    )
    all_pass &= check(
        calibfix_hparams["rank_extension"]["head_lr_multiplier"] == 10.0,
        "calibfix run (pre-revert) had rank_extension head_lr x10, confirming the revert is a real change",
    )

    print(f"\n{'=' * 100}\nOVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}\n{'=' * 100}")
    sys.exit(0 if all_pass else 1)
