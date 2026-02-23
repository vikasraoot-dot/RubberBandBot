import pytest
import numpy as np
import pandas as pd
from RubberBand.strategy import attach_verifiers, check_slope_filter

@pytest.fixture
def mock_df_flat():
    """Slope = 0.0% (Flat)"""
    return pd.DataFrame({
        "close": [100]*4,
        "kc_middle": [100.0, 100.0, 100.0, 100.0] 
    })

@pytest.fixture
def mock_df_crash():
    """Slope = -0.25% (Violent Crash)"""
    # 3-bar change needed: -0.75 on price 100
    # kc_middle: 100.75 -> 100.00
    return pd.DataFrame({
        "close": [100]*4,
        "kc_middle": [100.75, 100.50, 100.25, 100.00]
    })

@pytest.fixture
def mock_df_dip():
    """Slope = -0.15% (Moderate Dip)"""
    # 3-bar change needed: -0.45 on price 100
    return pd.DataFrame({
        "close": [100]*4,
        "kc_middle": [100.45, 100.30, 100.15, 100.00]
    })

def test_slope_calm_mode(mock_df_flat, mock_df_crash, mock_df_dip):
    """
    CALM Regime (Panic Buyer):
    - Expects Dip > 0.08
    - Flat (0.00) -> SKIP
    - Dip (-0.15) -> TRADE (It IS a dip)
    - Crash (-0.25) -> TRADE (It IS a dip)
    """
    regime_cfg = {"slope_threshold_pct": -0.08, "dead_knife_filter": False}
    
    # Flat
    skip, reason, _slope_pct = check_slope_filter(mock_df_flat, regime_cfg)
    assert skip is True
    assert "Too_Flat" in reason
    
    # Dip (-0.15 < -0.08) -> IS STEEP ENOUGH -> Trade
    skip, reason, _slope_pct = check_slope_filter(mock_df_dip, regime_cfg)
    assert skip is False
    
    # Crash (-0.25 < -0.08) -> IS STEEP ENOUGH -> Trade
    skip, reason, _slope_pct = check_slope_filter(mock_df_crash, regime_cfg)
    assert skip is False

def test_slope_panic_mode(mock_df_flat, mock_df_crash, mock_df_dip):
    """
    PANIC Regime (Safety Mode):
    - Expects Crash NOT > 0.20
    - Logic: SKIP if Slope < -0.20 (Too Steep)
    
    - Flat (0.00) -> TRADE (Safe)
    - Dip (-0.15) -> TRADE (Safe, > -0.20)
    - Crash (-0.25) -> SKIP (Unsafe, < -0.20)
    """
    regime_cfg = {"slope_threshold_pct": -0.20, "dead_knife_filter": True}
    
    # Flat (-0.00 > -0.20) -> SAFE
    skip, reason, _slope_pct = check_slope_filter(mock_df_flat, regime_cfg)
    assert skip is False
    
    # Dip (-0.15 > -0.20) -> SAFE
    skip, reason, _slope_pct = check_slope_filter(mock_df_dip, regime_cfg)
    assert skip is False
    
    # Crash (-0.25 < -0.20) -> UNSAFE -> SKIP
    skip, reason, _slope_pct = check_slope_filter(mock_df_crash, regime_cfg)
    assert skip is True
    assert "Safety_Knife_Filter" in reason


# ============================================================================
# Bearish Bar Filter Tests
# ============================================================================
from RubberBand.strategy import check_bearish_bar_filter


@pytest.fixture
def mock_df_bullish_bar():
    """Bar where close > open (bullish)"""
    return pd.DataFrame({
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [104.0],  # Close above open
    })


@pytest.fixture
def mock_df_bearish_bar():
    """Bar where close < open (bearish)"""
    return pd.DataFrame({
        "open": [104.0],
        "high": [105.0],
        "low": [99.0],
        "close": [100.0],  # Close below open
    })


@pytest.fixture
def mock_df_doji_bar():
    """Bar where close == open (doji)"""
    return pd.DataFrame({
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [100.0],  # Close equals open
    })


def test_bearish_bar_filter_disabled():
    """When filter is disabled, should never skip."""
    df = pd.DataFrame({
        "open": [104.0],
        "high": [105.0],
        "low": [99.0],
        "close": [100.0],  # Bearish bar
    })
    cfg = {"bearish_bar_filter": False}
    
    skip, reason = check_bearish_bar_filter(df, cfg)
    assert skip is False
    assert reason == ""


def test_bearish_bar_filter_bullish_bar(mock_df_bullish_bar):
    """Bullish bar (close > open) should NOT be skipped."""
    cfg = {"bearish_bar_filter": True}
    
    skip, reason = check_bearish_bar_filter(mock_df_bullish_bar, cfg)
    assert skip is False
    assert reason == ""


def test_bearish_bar_filter_bearish_bar(mock_df_bearish_bar):
    """Bearish bar (close < open) SHOULD be skipped."""
    cfg = {"bearish_bar_filter": True}
    
    skip, reason = check_bearish_bar_filter(mock_df_bearish_bar, cfg)
    assert skip is True
    assert "BearishBar_Filter" in reason
    assert "Close=100.00" in reason
    assert "Open=104.00" in reason


def test_bearish_bar_filter_doji(mock_df_doji_bar):
    """Doji bar (close == open) should NOT be skipped."""
    cfg = {"bearish_bar_filter": True}
    
    skip, reason = check_bearish_bar_filter(mock_df_doji_bar, cfg)
    assert skip is False
    assert reason == ""


def test_bearish_bar_filter_empty_df():
    """Empty DataFrame should not skip."""
    cfg = {"bearish_bar_filter": True}
    df = pd.DataFrame()
    
    skip, reason = check_bearish_bar_filter(df, cfg)
    assert skip is False


def test_bearish_bar_filter_missing_columns():
    """DataFrame missing open/close columns should not skip."""
    cfg = {"bearish_bar_filter": True}
    df = pd.DataFrame({"high": [105.0], "low": [99.0]})  # Missing open and close

    skip, reason = check_bearish_bar_filter(df, cfg)
    assert skip is False


# ============================================================================
# Bounce Confirmation Tests
# ============================================================================


def _make_ohlcv(n: int = 30, base: float = 100.0) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with a DatetimeIndex.

    Returns *n* bars of steady price action at *base*.  Callers modify
    the last few rows to engineer crash/bounce scenarios before passing
    the frame to ``attach_verifiers``.
    """
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min")
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.1, n)
    closes = base + noise
    df = pd.DataFrame(
        {
            "open": closes - 0.05,
            "high": closes + 0.30,
            "low": closes - 0.30,
            "close": closes,
            "volume": [100_000] * n,
        },
        index=idx,
    )
    return df


_BASE_CFG: dict = {
    "keltner_length": 20,
    "keltner_mult": 1.5,
    "atr_length": 14,
    "rsi_length": 14,
    "filters": {"rsi_oversold": 30, "rsi_min": 5},  # rsi_min=5 to not block deep crashes in tests
}


def _make_crash_df(n: int = 40, crash_bars: int = 5) -> pd.DataFrame:
    """Build OHLCV with a guaranteed multi-bar crash at the end.

    The last *crash_bars* bars drop progressively deeper from base 100,
    with the deepest point at bar -1.  Callers typically overwrite bar -1
    to create a bounce scenario.  The crash is deep enough to push RSI
    well below 30 and close below kc_lower.
    """
    df = _make_ohlcv(n, base=100.0)
    for i in range(-crash_bars, 0):
        # Progressive deepening: bar -crash_bars drops least, bar -1 drops most
        bar_num = crash_bars + i  # 0, 1, 2, ... crash_bars-1
        drop = 1.5 * (bar_num + 1)  # 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, ...
        df.iloc[i, df.columns.get_loc("close")] = 100.0 - drop
        df.iloc[i, df.columns.get_loc("open")] = 100.0 - drop + 1.0
        df.iloc[i, df.columns.get_loc("high")] = 100.0 - drop + 1.5
        df.iloc[i, df.columns.get_loc("low")] = 100.0 - drop - 0.5
    return df


def test_confirmation_bounce_bar_and_rsi_uptick():
    """Signal on bar N, bullish bar + rising RSI on bar N+1 -> confirmed."""
    df = _make_crash_df(40, crash_bars=6)

    # Overwrite bar -1 with a bounce bar (close > open, higher than prev close)
    df.iloc[-1, df.columns.get_loc("open")] = 92.5
    df.iloc[-1, df.columns.get_loc("close")] = 94.5
    df.iloc[-1, df.columns.get_loc("low")] = 92.0
    df.iloc[-1, df.columns.get_loc("high")] = 95.0

    cfg = {**_BASE_CFG, "confirmation": {"enabled": True}}
    result = attach_verifiers(df, cfg)

    # PRECONDITION: crash must have triggered raw signal on bar -2
    assert result.iloc[-2]["long_signal_raw"], (
        f"Precondition failed: no raw signal on bar -2. "
        f"RSI={result.iloc[-2]['rsi']:.1f}, "
        f"close={result.iloc[-2]['close']:.1f}, "
        f"kc_lower={result.iloc[-2]['kc_lower']:.1f}"
    )

    # Bounce bar conditions
    assert result.iloc[-1]["close"] > result.iloc[-1]["open"], "Bounce bar close > open"
    assert result.iloc[-1]["rsi"] > result.iloc[-2]["rsi"], "RSI should be rising"

    # Confirmation should fire
    assert result.iloc[-1]["long_signal_confirmed"] == True
    assert result.iloc[-1]["long_signal"] == True  # enabled=True overwrites


def test_confirmation_no_bounce_bar():
    """Signal on bar N, bearish continuation on N+1 -> NOT confirmed."""
    df = _make_crash_df(40, crash_bars=5)

    # Overwrite bar -1 with bearish continuation (close < open, still crashing)
    df.iloc[-1, df.columns.get_loc("open")] = 95.5
    df.iloc[-1, df.columns.get_loc("close")] = 94.0
    df.iloc[-1, df.columns.get_loc("low")] = 93.5
    df.iloc[-1, df.columns.get_loc("high")] = 95.8

    cfg = {**_BASE_CFG, "confirmation": {"enabled": True}}
    result = attach_verifiers(df, cfg)

    # Confirmation should NOT fire (bearish bar)
    assert result.iloc[-1]["long_signal_confirmed"] == False
    assert result.iloc[-1]["long_signal"] == False


def test_confirmation_no_rsi_uptick():
    """Bounce bar present but RSI still falling -> NOT confirmed.

    Tests the ``require_rsi_uptick`` condition in isolation by
    disabling ``require_bounce_bar``.
    """
    df = _make_crash_df(40, crash_bars=6)

    # Bar -1: tiny bounce (close > open) but close is still LOWER than
    # bar -2's close so RSI keeps dropping.
    prev_close = float(df.iloc[-2]["close"])
    df.iloc[-1, df.columns.get_loc("open")] = prev_close - 0.5
    df.iloc[-1, df.columns.get_loc("close")] = prev_close - 0.2  # close > open but < prev close
    df.iloc[-1, df.columns.get_loc("low")] = prev_close - 1.0
    df.iloc[-1, df.columns.get_loc("high")] = prev_close + 0.2

    cfg = {
        **_BASE_CFG,
        "confirmation": {
            "enabled": True,
            "require_bounce_bar": False,  # Isolate RSI uptick test
            "require_rsi_uptick": True,
        },
    }
    result = attach_verifiers(df, cfg)

    # RSI should still be falling (close dropped further)
    assert result.iloc[-1]["rsi"] <= result.iloc[-2]["rsi"], (
        f"Precondition failed: RSI should be falling. "
        f"bar -1 RSI={result.iloc[-1]['rsi']:.1f}, bar -2 RSI={result.iloc[-2]['rsi']:.1f}"
    )

    # Confirmation must NOT fire (RSI not rising)
    assert result.iloc[-1]["long_signal_confirmed"] == False


def test_confirmation_disabled():
    """When confirmation.enabled=false, long_signal equals long_signal_raw."""
    df = _make_crash_df(40, crash_bars=5)

    cfg = {**_BASE_CFG, "confirmation": {"enabled": False}}
    result = attach_verifiers(df, cfg)

    # long_signal must be identical to long_signal_raw
    pd.testing.assert_series_equal(
        result["long_signal"].astype(bool),
        result["long_signal_raw"].astype(bool),
        check_names=False,
    )
    # Columns should still exist even when disabled
    assert "long_signal_confirmed" in result.columns
    assert "long_signal_raw" in result.columns


def test_confirmation_multi_bar_persistence():
    """Signal persists across bars N, N+1.  Confirmation on N+1 bounce."""
    df = _make_crash_df(40, crash_bars=6)

    # Overwrite bar -1 with a strong bounce
    df.iloc[-1, df.columns.get_loc("open")] = 92.5
    df.iloc[-1, df.columns.get_loc("close")] = 95.5
    df.iloc[-1, df.columns.get_loc("low")] = 92.0
    df.iloc[-1, df.columns.get_loc("high")] = 96.0

    cfg = {**_BASE_CFG, "confirmation": {"enabled": True}}
    result = attach_verifiers(df, cfg)

    # PRECONDITION: bar -2 must have raw signal
    assert result.iloc[-2]["long_signal_raw"], (
        f"Precondition failed: no raw signal on bar -2. "
        f"RSI={result.iloc[-2]['rsi']:.1f}"
    )

    # Bounce should be confirmed
    assert result.iloc[-1]["close"] > result.iloc[-1]["open"]
    assert result.iloc[-1]["rsi"] > result.iloc[-2]["rsi"]
    assert result.iloc[-1]["long_signal_confirmed"] == True
    assert result.iloc[-1]["long_signal"] == True


def test_confirmation_signal_clears_before_bounce():
    """Signal on bar N, RSI recovers above threshold on N+1, bounce on N+2.

    Since long_signal_raw is False on bar N+1 (price recovered above
    kc_lower), the confirmation on bar N+2 should NOT fire.
    """
    df = _make_crash_df(40, crash_bars=4)

    # Bar -2: sharp V-recovery — price jumps back above Keltner, RSI recovers
    df.iloc[-2, df.columns.get_loc("open")] = 95.0
    df.iloc[-2, df.columns.get_loc("close")] = 100.0  # Back to normal
    df.iloc[-2, df.columns.get_loc("low")] = 94.8
    df.iloc[-2, df.columns.get_loc("high")] = 100.5

    # Bar -1: continuation (bullish, RSI rising further)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0
    df.iloc[-1, df.columns.get_loc("close")] = 101.0
    df.iloc[-1, df.columns.get_loc("low")] = 99.5
    df.iloc[-1, df.columns.get_loc("high")] = 101.5

    cfg = {**_BASE_CFG, "confirmation": {"enabled": True}}
    result = attach_verifiers(df, cfg)

    # PRECONDITION: bar -2 should NOT have raw signal (price recovered)
    assert not result.iloc[-2]["long_signal_raw"], (
        f"Precondition failed: bar -2 should have cleared. "
        f"close={result.iloc[-2]['close']:.1f}, "
        f"kc_lower={result.iloc[-2]['kc_lower']:.1f}, "
        f"RSI={result.iloc[-2]['rsi']:.1f}"
    )

    # Confirmation must NOT fire (prev bar had no raw signal)
    assert result.iloc[-1]["long_signal_confirmed"] == False


def test_confirmation_both_conditions_disabled():
    """When both require_bounce_bar and require_rsi_uptick are False,
    any bar following a raw signal should confirm (no conditions needed)."""
    df = _make_crash_df(40, crash_bars=5)

    # Bar -1: bearish continuation (would normally NOT confirm)
    df.iloc[-1, df.columns.get_loc("open")] = 96.0
    df.iloc[-1, df.columns.get_loc("close")] = 94.5
    df.iloc[-1, df.columns.get_loc("low")] = 94.0
    df.iloc[-1, df.columns.get_loc("high")] = 96.5

    cfg = {
        **_BASE_CFG,
        "confirmation": {
            "enabled": True,
            "require_bounce_bar": False,
            "require_rsi_uptick": False,
        },
    }
    result = attach_verifiers(df, cfg)

    # With both conditions disabled, confirmation = just prev bar had signal
    if result.iloc[-2]["long_signal_raw"]:
        assert result.iloc[-1]["long_signal_confirmed"] == True

