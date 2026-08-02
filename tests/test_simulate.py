"""Simulator determinism, mode separation, and schema validity."""

import pytest

from longshot.bias import compression_slope
from longshot.horizons import build_panel, parse_horizon
from longshot.simulate import simulate_markets
from longshot.store import market_from_dict, market_to_dict


def test_determinism_by_seed():
    a = simulate_markets("compressed", 10, 42)
    b = simulate_markets("compressed", 10, 42)
    c = simulate_markets("compressed", 10, 43)
    assert [market_to_dict(m) for m in a] == [market_to_dict(m) for m in b]
    assert [market_to_dict(m) for m in a] != [market_to_dict(m) for m in c]


def test_modes_produce_different_slopes():
    kwargs = dict(min_per_bin=20, n_boot=100, seed=1)
    cal = compression_slope(
        build_panel(simulate_markets("calibrated", 300, 7),
                    parse_horizon("30d"), "30d"), **kwargs)
    comp = compression_slope(
        build_panel(simulate_markets("compressed", 300, 11),
                    parse_horizon("30d"), "30d"), **kwargs)
    assert comp["slope"] - cal["slope"] > 0.5


def test_schema_validates_via_store():
    for mode in ("calibrated", "compressed", "longshot-bias"):
        for m in simulate_markets(mode, 5, 13):
            round_tripped = market_from_dict(market_to_dict(m))
            assert round_tripped is not None
            assert round_tripped.market_id == m.market_id
            assert round_tripped.venue == "simulated"
            assert round_tripped.provenance["mode"] == mode


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        simulate_markets("not-a-mode", 1, 1)


def test_terminal_convergence():
    # Final price sits at the outcome side of the book.
    for m in simulate_markets("calibrated", 20, 21):
        last = m.series[-1].price
        assert last == pytest.approx(0.99 if m.outcome else 0.01)
