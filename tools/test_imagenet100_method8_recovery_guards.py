"""Dry test for normal and Method-8-only active-arm contracts."""

from __future__ import annotations

try:
    from tools.imagenet100_method_guards import assert_active_method_contract
except ModuleNotFoundError:
    from imagenet100_method_guards import assert_active_method_contract


METHOD8 = "rank_extension_factor_orth_lam50_fullkd_T2_protect30"
METHOD6 = "rank_extension_fullkd_T2_protect30"
METHOD7 = "rank_extension_factor_orth_lam50_new"


def _map(*names: str) -> dict[str, dict[str, object]]:
    return {name: {"family": "rank_extension"} for name in names}


def main() -> None:
    assert_active_method_contract(
        [METHOD6, METHOD7, METHOD8], _map(METHOD6, METHOD7, METHOD8),
        recovery_mode=False, method8_name=METHOD8,
    )
    assert_active_method_contract(
        [METHOD8], _map(METHOD8), recovery_mode=True, method8_name=METHOD8,
    )
    try:
        assert_active_method_contract(
            [METHOD6, METHOD8], _map(METHOD6, METHOD8),
            recovery_mode=True, method8_name=METHOD8,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("recovery guard accepted an extra RankExt arm")
    print("PASS: normal 3-arm and Method-8-only recovery contracts")


if __name__ == "__main__":
    main()
