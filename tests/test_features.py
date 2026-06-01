"""Smoke tests for the feature engineering module.

Uses synthetic orderbook data so tests run without LOBSTER samples available.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    order_flow_imbalance,
    volume_imbalance,
    cumulative_depth_imbalance,
    mid_price_return,
    micro_minus_mid,
    trade_flow_imbalance,
    build_feature_panel,
    make_targets,
)


@pytest.fixture
def synthetic_book():
    """5-row synthetic level-1 orderbook with deliberate patterns."""
    return pd.DataFrame({
        "ask_price_1": [100.10, 100.10, 100.20, 100.20, 100.30],
        "ask_size_1": [50, 30, 100, 80, 60],
        "bid_price_1": [100.00, 100.05, 100.05, 100.10, 100.10],
        "bid_size_1": [100, 80, 90, 70, 110],
    })


@pytest.fixture
def synthetic_messages():
    return pd.DataFrame({
        "time": [0.0, 0.5, 1.0, 1.5, 2.0],
        "type": [1, 4, 1, 4, 2],
        "order_id": [1, 1, 2, 2, 3],
        "size": [50, 30, 40, 20, 10],
        "price": [100.10, 100.05, 100.20, 100.10, 100.30],
        "direction": [-1, 1, -1, 1, 1],
        "type_name": ["submission", "execution_visible", "submission", "execution_visible", "cancellation_partial"],
    })


def test_ofi_shape(synthetic_book):
    ofi = order_flow_imbalance(synthetic_book, level=1)
    assert len(ofi) == len(synthetic_book)
    assert ofi.iloc[0] == 0  # first event has no previous reference


def test_ofi_buying_pressure(synthetic_book):
    """Row 1->2: bid_price rises 100.00 -> 100.05, ask_size drops 50 -> 30.
    Both signal buying pressure => OFI should be positive."""
    ofi = order_flow_imbalance(synthetic_book, level=1)
    assert ofi.iloc[1] > 0


def test_volume_imbalance_range(synthetic_book):
    vi = volume_imbalance(synthetic_book, level=1)
    assert ((vi >= -1) & (vi <= 1)).all()


def test_cumulative_depth_imbalance(synthetic_book):
    # Only level 1 in synthetic data — should equal vol_imb_L1
    cdi = cumulative_depth_imbalance(synthetic_book, max_level=5)
    vi = volume_imbalance(synthetic_book, level=1)
    np.testing.assert_array_almost_equal(cdi.values, vi.values)


def test_mid_price_return_horizon(synthetic_book):
    ret = mid_price_return(synthetic_book, horizon=1)
    # mid: 100.05, 100.075, 100.125, 100.15, 100.20
    # ret_h1 = log(mid[t+1] / mid[t]); last value is NaN
    assert pd.isna(ret.iloc[-1])
    assert ret.iloc[0] == pytest.approx(np.log(100.075 / 100.05))


def test_trade_flow_imbalance(synthetic_messages):
    flow = trade_flow_imbalance(synthetic_messages, rolling_seconds=1.0)
    assert len(flow) == len(synthetic_messages)
    # At t=0.5, the only event in [-0.5, 0.5] is the buy execution of size 30
    assert flow.iloc[1] == pytest.approx(30.0)


def test_build_feature_panel(synthetic_book, synthetic_messages):
    features = build_feature_panel(synthetic_messages, synthetic_book, ofi_max_level=1)
    assert "ofi_L1" in features.columns
    assert "vol_imb_L1" in features.columns
    assert "micro_dev" in features.columns
    assert "spread_ticks" in features.columns
    assert len(features) == len(synthetic_book)


def test_make_targets(synthetic_book):
    targets = make_targets(synthetic_book, horizons=(1, 3))
    assert "target_dir_h1" in targets.columns
    assert "target_dir_h3" in targets.columns
    # h=1: future mid > current mid in all 4 transitions => all +1 (except last NaN)
    assert (targets["target_dir_h1"].dropna() == 1).all()
