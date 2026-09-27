"""Fail-closed active-method contracts for ImageNet continuation modes."""

from __future__ import annotations


NORMAL_CONTINUATION_METHODS = {
    "rank_extension_fullkd_T2_protect30",
    "rank_extension_factor_orth_lam50_new",
    "rank_extension_factor_orth_lam50_fullkd_T2_protect30",
}


def assert_active_method_contract(
    active_method_names: list[str],
    active_method_map: dict[str, dict[str, object]],
    *,
    recovery_mode: bool,
    method8_name: str,
) -> None:
    """Validate the exact active-arm contract before any training starts."""
    simple_avg_arms = [
        name for name in active_method_names
        if active_method_map[name]["family"] == "simple_avg"
    ]
    rankext_arms = [
        name for name in active_method_names
        if active_method_map[name]["family"] == "rank_extension"
    ]
    if recovery_mode:
        assert len(simple_avg_arms) == 0, "Method-8 recovery cannot activate SimpleAvg"
        assert len(rankext_arms) == 1, "Method-8 recovery requires exactly one RankExt arm"
        assert active_method_names == [method8_name], (
            "Method-8 recovery active methods must be exactly "
            f"[{method8_name!r}], got {active_method_names!r}"
        )
    else:
        assert len(simple_avg_arms) == 0 and len(rankext_arms) == 3 and len(active_method_names) == 3
        assert set(active_method_names) == NORMAL_CONTINUATION_METHODS
