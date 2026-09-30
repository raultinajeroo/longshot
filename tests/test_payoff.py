"""Quoted payoffs use asks, explicit costs, and the held-out predictions."""

from dataclasses import replace

import numpy as np
import pytest

from longshot.correct import run_correction
from longshot.simulate import simulate_markets
from longshot.types import HorizonPoint


def quoted_points():
    return [HorizonPoint(
        market_id=f"m{i}", category="test", volume=None,
        resolved_ts=(100 + i) * 86400, observed_ts=(99 + i) * 86400,
        p=0.5, outcome=int(i < 15), bid=0.4, ask=0.6,
    ) for i in range(30)]


def screen(points, predictions, cost=0.05, groups=None):
    from longshot.correct import _quoted_payoff

    return _quoted_payoff(
        points, np.asarray(predictions),
        groups if groups is not None else [[i] for i in range(len(points))],
        cost_per_contract=cost, n_boot=100, seed=42,
    )


def test_payoff_uses_ask_and_keeps_abstentions_in_the_denominator():
    points = quoted_points()
    result = screen(points, [0.8] * 20 + [0.5] * 10)
    # Twenty selected: 15 wins at +0.35, five losses at -0.65. The other
    # ten quoted markets contribute zero; midpoints would overstate profit.
    assert result["n_quoted"] == 30
    assert result["n_selected"] == 20
    assert result["total_payoff"] == pytest.approx(2.0)
    assert result["mean_payoff"] == pytest.approx(2 / 30)
    assert result["n_groups"] == 30
    changed_labels = [replace(p, outcome=1 - p.outcome) for p in points]
    assert screen(changed_labels, [0.8] * 20 + [0.5] * 10)["n_selected"] == 20


def test_cost_can_remove_every_candidate_and_equality_does_not_qualify():
    result = screen(quoted_points(), [0.8] * 30, cost=0.2)
    assert result["n_selected"] == 0
    assert result["total_payoff"] == 0
    assert result["mean_payoff_ci"] == [0, 0]


def test_missing_quotes_are_excluded_instead_of_replaced_with_midpoints():
    points = quoted_points() + [replace(quoted_points()[0], bid=None, ask=None)]
    result = screen(points, [0.8] * 31)
    assert result["n_quoted"] == 30
    assert result["n_missing_quotes"] == 1
    assert result["n_selected"] == 30
    legacy = [replace(p, bid=None, ask=None) for p in points]
    result = screen(legacy, [0.8] * 31)
    assert result["n_quoted"] == 0
    assert "too few quoted test points" in result["skipped"]
    assert "mean_payoff" not in result


def test_one_resolution_group_cannot_support_payoff_interval():
    result = screen(quoted_points(), [0.8] * 30, groups=[list(range(30))])
    assert result["n_groups"] == 1
    assert "at least 2 quoted test groups" in result["skipped"]


def test_cost_screen_leaves_calibration_and_time_cut_unchanged():
    markets = simulate_markets("calibrated", 400, 202)
    markets = [replace(m, series=tuple(
        replace(p, bid=max(0, p.price - 0.02), ask=min(1, p.price + 0.02))
        for p in m.series)) for m in markets]
    args = dict(horizons=["30d"], horizon_seconds={"30d": 30 * 86400}, n_boot=30)
    original = run_correction(markets, **args)
    costed = run_correction(markets, **args, cost_per_contract=0.01)
    for method in ("platt", "isotonic"):
        old, new = (r["horizons"]["30d"][method] for r in (original, costed))
        assert new["brier"] == old["brier"]
        assert new["delta_brier_ci"] == old["delta_brier_ci"]
        assert new["quoted_payoff"]["n_quoted"] == costed["horizons"]["30d"]["n_test"]
    assert costed["horizons"]["30d"]["n_train_purged"] == original["horizons"]["30d"]["n_train_purged"]


@pytest.mark.parametrize("cost", [-0.1, float("nan"), float("inf")])
def test_cost_must_be_finite_and_nonnegative(cost):
    with pytest.raises(ValueError, match="cost per contract"):
        run_correction(simulate_markets("calibrated", 30, 1), horizons=["1d"],
                       horizon_seconds={"1d": 86400}, cost_per_contract=cost)
