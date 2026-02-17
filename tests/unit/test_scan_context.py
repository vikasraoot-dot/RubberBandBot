"""
Tests for Signal Context Logging (SCAN_CONTEXT).

Covers:
- SlopeFilterResult NamedTuple
- TradeLogger.scan_context() emitter
- build_symbol_context() helper
- _safe_float_ctx() sanitizer
- Schema validation
- Edge cases (NaN, numpy types, empty universe)
"""
import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from RubberBand.strategy import check_slope_filter, SlopeFilterResult
from RubberBand.src.trade_logger import TradeLogger
from RubberBand.scripts.live_paper_loop import (
    build_symbol_context,
    _safe_float_ctx,
    SCAN_OUTCOME_NO_DATA,
    SCAN_OUTCOME_SIGNAL,
    SCAN_OUTCOME_SKIP_SLOPE,
    SCAN_OUTCOME_NO_SIGNAL,
    SCAN_OUTCOME_INSUFFICIENT_BARS,
    SCAN_OUTCOME_SKIP_BEARISH,
    SCAN_OUTCOME_DKF_SKIP,
)


# ============================================================================
# _safe_float_ctx tests
# ============================================================================

class TestSafeFloatCtx:
    def test_normal_float(self):
        assert _safe_float_ctx(3.14159) == 3.1416

    def test_numpy_float64(self):
        val = np.float64(142.5678)
        result = _safe_float_ctx(val)
        assert isinstance(result, float)
        assert result == 142.5678

    def test_numpy_int64(self):
        val = np.int64(42)
        result = _safe_float_ctx(val)
        assert isinstance(result, float)
        assert result == 42.0

    def test_nan_returns_none(self):
        assert _safe_float_ctx(float('nan')) is None

    def test_inf_returns_none(self):
        assert _safe_float_ctx(float('inf')) is None

    def test_neg_inf_returns_none(self):
        assert _safe_float_ctx(float('-inf')) is None

    def test_numpy_nan_returns_none(self):
        assert _safe_float_ctx(np.nan) is None

    def test_none_returns_none(self):
        assert _safe_float_ctx(None) is None

    def test_string_returns_none(self):
        assert _safe_float_ctx("not_a_number") is None

    def test_zero(self):
        assert _safe_float_ctx(0.0) == 0.0

    def test_custom_decimals(self):
        assert _safe_float_ctx(3.14159, decimals=2) == 3.14


# ============================================================================
# SlopeFilterResult NamedTuple tests
# ============================================================================

class TestSlopeFilterResult:
    def test_returns_namedtuple(self):
        result = check_slope_filter(pd.DataFrame({"close": [100]*4, "kc_middle": [100]*4}),
                                     {"slope_threshold_pct": -0.08, "dead_knife_filter": False})
        assert isinstance(result, SlopeFilterResult)

    def test_3tuple_unpack(self):
        """Verify 3-value destructuring works."""
        df = pd.DataFrame({"close": [100]*4, "kc_middle": [100]*4})
        skip, reason, slope_pct = check_slope_filter(df, {"slope_threshold_pct": -0.08, "dead_knife_filter": False})
        assert isinstance(skip, bool)
        assert isinstance(reason, str)
        assert isinstance(slope_pct, float)

    def test_slope_pct_on_skip(self):
        """Slope pct is propagated when filter skips."""
        # Flat market -> skip in CALM mode
        df = pd.DataFrame({"close": [100]*4, "kc_middle": [100.0, 100.0, 100.0, 100.0]})
        result = check_slope_filter(df, {"slope_threshold_pct": -0.08, "dead_knife_filter": False})
        assert result.should_skip is True
        assert result.slope_pct == 0.0  # Flat = 0%

    def test_slope_pct_on_pass(self):
        """Slope pct is propagated when filter passes."""
        # Dip -> pass in CALM mode
        df = pd.DataFrame({"close": [100]*4, "kc_middle": [100.45, 100.30, 100.15, 100.00]})
        result = check_slope_filter(df, {"slope_threshold_pct": -0.08, "dead_knife_filter": False})
        assert result.should_skip is False
        assert result.slope_pct < 0  # Negative slope

    def test_slope_pct_insufficient_data(self):
        """Returns 0.0 slope when insufficient data."""
        df = pd.DataFrame({"close": [100]*2, "kc_middle": [100]*2})
        result = check_slope_filter(df, {})
        assert result.should_skip is False
        assert result.slope_pct == 0.0

    def test_slope_pct_no_kc_middle(self):
        """Returns 0.0 slope when kc_middle column missing."""
        df = pd.DataFrame({"close": [100]*4})
        result = check_slope_filter(df, {})
        assert result.should_skip is False
        assert result.slope_pct == 0.0

    def test_zero_close_price(self):
        """Handles zero close price without division error."""
        df = pd.DataFrame({"close": [0]*4, "kc_middle": [100]*4})
        result = check_slope_filter(df, {})
        assert result.should_skip is False
        assert result.slope_pct == 0.0

    def test_named_access(self):
        """Can access fields by name."""
        df = pd.DataFrame({"close": [100]*4, "kc_middle": [100]*4})
        result = check_slope_filter(df, {"slope_threshold_pct": -0.08, "dead_knife_filter": False})
        assert hasattr(result, 'should_skip')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'slope_pct')

    def test_panic_mode_skip(self):
        """PANIC mode: steep crash triggers skip with slope_pct."""
        df = pd.DataFrame({"close": [100]*4, "kc_middle": [100.75, 100.50, 100.25, 100.00]})
        result = check_slope_filter(df, {"slope_threshold_pct": -0.20, "dead_knife_filter": True})
        assert result.should_skip is True
        assert "Safety_Knife_Filter" in result.reason
        assert result.slope_pct < -0.20


# ============================================================================
# build_symbol_context tests
# ============================================================================

class TestBuildSymbolContext:
    def test_minimal_no_data(self):
        """Early exit symbols get minimal snapshot."""
        ctx = build_symbol_context("AAPL", SCAN_OUTCOME_NO_DATA, filters={"data_ok": False})
        assert ctx["ticker"] == "AAPL"
        assert ctx["outcome"] == "NO_DATA"
        assert ctx["filters"]["data_ok"] is False
        assert "close" not in ctx
        assert "rsi" not in ctx

    def test_full_snapshot(self):
        """Symbols reaching signals get full indicator set."""
        ctx = build_symbol_context(
            "NVDA", SCAN_OUTCOME_SIGNAL,
            close=142.5, open_price=140.0, rsi=22.4, atr=3.12,
            kc_lower=138.2, kc_middle=145.6, kc_upper=153.0,
            slope_pct=-0.31, is_bull_trend=True, is_strong_bull=True,
            dollar_vol=2500000.0, rvol=1.5, gap_pct=-2.3, bars_count=45,
            filters={"data_ok": True, "signal": True},
        )
        assert ctx["ticker"] == "NVDA"
        assert ctx["outcome"] == "SIGNAL"
        assert ctx["close"] == 142.5
        assert ctx["open"] == 140.0
        assert ctx["rsi"] == 22.4
        assert ctx["kc_middle"] == 145.6
        assert ctx["slope_pct"] == -0.31
        assert ctx["is_bull_trend"] is True
        assert ctx["is_strong_bull"] is True
        assert ctx["rvol"] == 1.5
        assert ctx["gap_pct"] == -2.3
        assert ctx["bars_count"] == 45

    def test_none_values_excluded(self):
        """None values are not included in context dict."""
        ctx = build_symbol_context("TEST", SCAN_OUTCOME_NO_SIGNAL, close=100.0, rsi=None)
        assert "close" in ctx
        assert "rsi" not in ctx

    def test_nan_values_excluded(self):
        """NaN values are sanitized out."""
        ctx = build_symbol_context("TEST", SCAN_OUTCOME_SKIP_SLOPE, close=100.0, rsi=float('nan'))
        assert "rsi" not in ctx
        assert "close" in ctx

    def test_numpy_types_converted(self):
        """Numpy types are converted to native Python types."""
        ctx = build_symbol_context(
            "TEST", SCAN_OUTCOME_SIGNAL,
            close=np.float64(142.5),
            rsi=np.float64(22.4),
            atr=np.float64(3.12),
        )
        assert isinstance(ctx["close"], float)
        assert isinstance(ctx["rsi"], float)
        assert isinstance(ctx["atr"], float)

    def test_json_serializable(self):
        """Result is fully JSON-serializable with correct types."""
        ctx = build_symbol_context(
            "TEST", SCAN_OUTCOME_SIGNAL,
            close=np.float64(142.5), rsi=np.float64(22.4),
            is_bull_trend=True, bars_count=45,
            filters={"data_ok": True, "signal": True},
        )
        serialized = json.dumps(ctx)
        deserialized = json.loads(serialized)
        # Verify types survived serialization
        assert isinstance(deserialized["close"], float)
        assert isinstance(deserialized["is_bull_trend"], bool)
        assert isinstance(deserialized["bars_count"], int)
        assert deserialized["close"] == 142.5  # Not string "142.5"

    def test_batch_structure(self):
        """Multiple symbols form a valid batch array."""
        batch = [
            build_symbol_context("AAPL", SCAN_OUTCOME_NO_DATA, filters={"data_ok": False}),
            build_symbol_context("NVDA", SCAN_OUTCOME_SIGNAL, close=142.5, rsi=22.4,
                                 filters={"signal": True}),
            build_symbol_context("MSFT", SCAN_OUTCOME_SKIP_SLOPE, close=400.0,
                                 slope_pct=-0.05, filters={"slope_ok": False}),
        ]
        assert len(batch) == 3
        serialized = json.dumps(batch)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 3
        assert deserialized[0]["outcome"] == "NO_DATA"
        assert deserialized[1]["outcome"] == "SIGNAL"
        assert deserialized[2]["outcome"] == "SKIP_SLOPE"


# ============================================================================
# TradeLogger.scan_context() tests
# ============================================================================

class TestScanContextEmitter:
    def test_writes_valid_jsonl(self):
        """scan_context writes valid JSONL to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir='.') as f:
            path = f.name

        try:
            logger = TradeLogger(path)
            logger.scan_context(
                schema_v=1,
                regime="NORMAL",
                symbols_scanned=2,
                symbols_passed=1,
                symbols=[
                    {"ticker": "AAPL", "outcome": "NO_DATA"},
                    {"ticker": "NVDA", "outcome": "SIGNAL", "close": 142.5},
                ],
            )
            logger.close()

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            assert len(lines) == 1
            event = json.loads(lines[0])
            assert event["type"] == "SCAN_CONTEXT"
            assert event["schema_v"] == 1
            assert event["regime"] == "NORMAL"
            assert len(event["symbols"]) == 2
            assert "ts" in event
            assert "ts_et" in event
        finally:
            os.unlink(path)

    def test_does_not_mirror_to_stdout(self, capsys):
        """SCAN_CONTEXT should NOT be printed to stdout (too large)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir='.') as f:
            path = f.name

        try:
            logger = TradeLogger(path)
            logger.scan_context(
                symbols_scanned=1,
                symbols=[{"ticker": "TEST", "outcome": "SIGNAL"}],
            )
            logger.close()

            captured = capsys.readouterr()
            assert "SCAN_CONTEXT" not in captured.out
        finally:
            os.unlink(path)

    def test_schema_required_fields(self):
        """SCAN_CONTEXT events have required fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir='.') as f:
            path = f.name

        try:
            logger = TradeLogger(path)
            logger.scan_context(
                schema_v=1,
                regime="CALM",
                regime_detail={"vixy_price": 20.0},
                market_context={"spy_change_pct": 0.5},
                portfolio_context={"open_positions_count": 2},
                symbols_scanned=100,
                symbols_passed=3,
                symbols=[],
            )
            logger.close()

            with open(path, 'r', encoding='utf-8') as f:
                event = json.loads(f.readline())

            # Verify all required fields present
            assert event["type"] == "SCAN_CONTEXT"
            assert event["schema_v"] == 1
            assert event["regime"] == "CALM"
            assert "regime_detail" in event
            assert "market_context" in event
            assert "portfolio_context" in event
            assert event["symbols_scanned"] == 100
            assert event["symbols_passed"] == 3
            assert event["symbols"] == []
        finally:
            os.unlink(path)


# ============================================================================
# Outcome code vocabulary tests
# ============================================================================

class TestOutcomeCodes:
    def test_all_outcome_codes_defined(self):
        """All outcome codes are defined as constants."""
        from RubberBand.scripts.live_paper_loop import (
            SCAN_OUTCOME_NO_DATA,
            SCAN_OUTCOME_INSUFFICIENT_BARS,
            SCAN_OUTCOME_PAUSED_HEALTH,
            SCAN_OUTCOME_FORMING_DROPPED,
            SCAN_OUTCOME_TREND_NO_DATA,
            SCAN_OUTCOME_TREND_BEAR,
            SCAN_OUTCOME_SKIP_SLOPE,
            SCAN_OUTCOME_SKIP_BEARISH,
            SCAN_OUTCOME_DKF_SKIP,
            SCAN_OUTCOME_NO_SIGNAL,
            SCAN_OUTCOME_ALREADY_IN,
            SCAN_OUTCOME_TRADED_TODAY,
            SCAN_OUTCOME_BAD_TP_SL,
            SCAN_OUTCOME_QTY_ZERO,
            SCAN_OUTCOME_SIGNAL,
        )
        # All are non-empty strings
        codes = [
            SCAN_OUTCOME_NO_DATA, SCAN_OUTCOME_INSUFFICIENT_BARS,
            SCAN_OUTCOME_PAUSED_HEALTH, SCAN_OUTCOME_FORMING_DROPPED,
            SCAN_OUTCOME_TREND_NO_DATA, SCAN_OUTCOME_TREND_BEAR,
            SCAN_OUTCOME_SKIP_SLOPE, SCAN_OUTCOME_SKIP_BEARISH,
            SCAN_OUTCOME_DKF_SKIP, SCAN_OUTCOME_NO_SIGNAL,
            SCAN_OUTCOME_ALREADY_IN, SCAN_OUTCOME_TRADED_TODAY,
            SCAN_OUTCOME_BAD_TP_SL, SCAN_OUTCOME_QTY_ZERO,
            SCAN_OUTCOME_SIGNAL,
        ]
        for code in codes:
            assert isinstance(code, str) and len(code) > 0

    def test_outcome_codes_unique(self):
        """All outcome codes are unique."""
        from RubberBand.scripts import live_paper_loop as lpl
        codes = [v for k, v in vars(lpl).items() if k.startswith("SCAN_OUTCOME_")]
        assert len(codes) == len(set(codes))


# ============================================================================
# Large batch serialization roundtrip
# ============================================================================

class TestSerializationRoundtrip:
    def test_200_symbol_batch(self):
        """200-symbol batch serializes and deserializes with correct types."""
        symbols = []
        for i in range(200):
            if i < 50:
                symbols.append(build_symbol_context(f"SYM{i}", SCAN_OUTCOME_NO_DATA,
                                                     filters={"data_ok": False}))
            elif i < 150:
                symbols.append(build_symbol_context(
                    f"SYM{i}", SCAN_OUTCOME_NO_SIGNAL,
                    close=float(100 + i * 0.5),
                    rsi=float(30 + i * 0.1),
                    slope_pct=float(-0.1 + i * 0.001),
                    filters={"signal": False},
                ))
            else:
                symbols.append(build_symbol_context(
                    f"SYM{i}", SCAN_OUTCOME_SIGNAL,
                    close=np.float64(100 + i),
                    rsi=np.float64(22.0),
                    atr=np.float64(3.0),
                    kc_lower=np.float64(97.0),
                    kc_middle=np.float64(100.0),
                    kc_upper=np.float64(103.0),
                    is_bull_trend=True,
                    is_strong_bull=True,
                    filters={"signal": True},
                ))

        event = {
            "type": "SCAN_CONTEXT",
            "schema_v": 1,
            "symbols_scanned": 200,
            "symbols": symbols,
        }

        serialized = json.dumps(event, default=str)
        deserialized = json.loads(serialized)

        assert len(deserialized["symbols"]) == 200
        # Verify numeric types survived (not stringified)
        signal_sym = deserialized["symbols"][150]
        assert isinstance(signal_sym["close"], (int, float))
        assert isinstance(signal_sym["rsi"], (int, float))
        assert signal_sym["is_bull_trend"] is True


# ============================================================================
# Regression tests
# ============================================================================

class TestRegressions:
    def test_filter_diagnostics_still_emitted(self):
        """FILTER_DIAGNOSTICS event type still exists in live_paper_loop.py."""
        import ast
        loop_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "RubberBand", "scripts", "live_paper_loop.py"
        )
        with open(loop_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert '"FILTER_DIAGNOSTICS"' in source, \
            "REGRESSION: FILTER_DIAGNOSTICS event must not be removed"

    def test_existing_slope_filter_tests_pass_with_3tuple(self):
        """Existing slope filter behavior unchanged with NamedTuple."""
        # CALM mode: flat -> skip
        df_flat = pd.DataFrame({"close": [100]*4, "kc_middle": [100.0]*4})
        skip, reason, slope_pct = check_slope_filter(
            df_flat, {"slope_threshold_pct": -0.08, "dead_knife_filter": False})
        assert skip is True
        assert "Too_Flat" in reason

        # CALM mode: dip -> pass
        df_dip = pd.DataFrame({"close": [100]*4, "kc_middle": [100.45, 100.30, 100.15, 100.00]})
        skip, reason, slope_pct = check_slope_filter(
            df_dip, {"slope_threshold_pct": -0.08, "dead_knife_filter": False})
        assert skip is False

        # PANIC mode: crash -> skip
        df_crash = pd.DataFrame({"close": [100]*4, "kc_middle": [100.75, 100.50, 100.25, 100.00]})
        skip, reason, slope_pct = check_slope_filter(
            df_crash, {"slope_threshold_pct": -0.20, "dead_knife_filter": True})
        assert skip is True
        assert "Safety_Knife_Filter" in reason

    def test_slope_filter_importable_from_strategy(self):
        """SlopeFilterResult is importable from strategy module."""
        from RubberBand.strategy import SlopeFilterResult
        assert hasattr(SlopeFilterResult, '_fields')
        assert SlopeFilterResult._fields == ('should_skip', 'reason', 'slope_pct')
