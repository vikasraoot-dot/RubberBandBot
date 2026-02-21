"""
Regression tests for options_data IV extraction.

Verifies that implied volatility is read from the top-level snapshot field
(impliedVolatility / implied_volatility), NOT from the greeks sub-object.
This bug caused IV to always be 0.0, breaking the probability filter's
Alpaca IV fallback path.
"""
import pytest
from unittest.mock import patch, MagicMock

from RubberBand.src.options_data import (
    get_option_snapshot,
    get_option_snapshots_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_api_response(snapshots: dict) -> MagicMock:
    """Build a mock requests.get() return value for the snapshots endpoint."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"snapshots": snapshots}
    return resp


def _base_snapshot(*, iv_key="impliedVolatility", iv_val=0.32):
    """Alpaca-style raw snapshot with IV at the top level (not inside greeks)."""
    snap = {
        "latestQuote": {"bp": 1.10, "ap": 1.30},
        "greeks": {
            "delta": 0.55,
            "theta": -0.08,
            "gamma": 0.04,
            "vega": 0.15,
            # NOTE: no implied_volatility inside greeks — that's the point
        },
    }
    if iv_key and iv_val is not None:
        snap[iv_key] = iv_val
    return snap


SYM = "AAPL260220C00230000"
CREDS_PATCH = "RubberBand.src.options_data._resolve_creds"
REQ_PATCH = "RubberBand.src.options_data.requests.get"


# ---------------------------------------------------------------------------
# get_option_snapshot — single symbol
# ---------------------------------------------------------------------------
class TestGetOptionSnapshotIV:
    """IV extraction from single-symbol snapshot."""

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_iv_from_camel_case_top_level(self, mock_get, _creds):
        """impliedVolatility (camelCase) at top level is extracted."""
        mock_get.return_value = _mock_api_response(
            {SYM: _base_snapshot(iv_key="impliedVolatility", iv_val=0.32)}
        )
        result = get_option_snapshot(SYM)
        assert result is not None
        assert result["iv"] == pytest.approx(0.32, abs=1e-6)

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_iv_from_snake_case_top_level(self, mock_get, _creds):
        """implied_volatility (snake_case) at top level is extracted."""
        mock_get.return_value = _mock_api_response(
            {SYM: _base_snapshot(iv_key="implied_volatility", iv_val=0.28)}
        )
        result = get_option_snapshot(SYM)
        assert result is not None
        assert result["iv"] == pytest.approx(0.28, abs=1e-6)

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_iv_zero_when_missing(self, mock_get, _creds):
        """IV defaults to 0.0 when absent from snapshot entirely."""
        snap = _base_snapshot()
        # Remove IV from top level
        snap.pop("impliedVolatility", None)
        snap.pop("implied_volatility", None)
        mock_get.return_value = _mock_api_response({SYM: snap})
        result = get_option_snapshot(SYM)
        assert result is not None
        assert result["iv"] == 0.0

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_iv_none_from_api_returns_zero(self, mock_get, _creds):
        """impliedVolatility: null in JSON -> iv: 0.0."""
        mock_get.return_value = _mock_api_response(
            {SYM: _base_snapshot(iv_key="impliedVolatility", iv_val=None)}
        )
        result = get_option_snapshot(SYM)
        assert result is not None
        assert result["iv"] == 0.0

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_negative_iv_clamped_to_zero(self, mock_get, _creds):
        """Negative IV (data error) is clamped to 0.0."""
        mock_get.return_value = _mock_api_response(
            {SYM: _base_snapshot(iv_key="impliedVolatility", iv_val=-0.5)}
        )
        result = get_option_snapshot(SYM)
        assert result is not None
        assert result["iv"] == 0.0

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_greeks_still_extracted_correctly(self, mock_get, _creds):
        """Delta, theta, gamma, vega still come from greeks sub-object."""
        mock_get.return_value = _mock_api_response(
            {SYM: _base_snapshot(iv_key="impliedVolatility", iv_val=0.32)}
        )
        result = get_option_snapshot(SYM)
        assert result is not None
        assert result["delta"] == pytest.approx(0.55)
        assert result["theta"] == pytest.approx(-0.08)
        assert result["gamma"] == pytest.approx(0.04)
        assert result["vega"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# get_option_snapshots_batch — multiple symbols
# ---------------------------------------------------------------------------
class TestGetOptionSnapshotsBatchIV:
    """IV extraction from batch snapshot endpoint."""

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_batch_iv_from_top_level(self, mock_get, _creds):
        """Batch endpoint extracts IV from top-level impliedVolatility."""
        sym2 = "AAPL260220P00230000"
        mock_get.return_value = _mock_api_response({
            SYM: _base_snapshot(iv_key="impliedVolatility", iv_val=0.32),
            sym2: _base_snapshot(iv_key="impliedVolatility", iv_val=0.45),
        })
        result = get_option_snapshots_batch([SYM, sym2])
        assert result[SYM]["iv"] == pytest.approx(0.32, abs=1e-6)
        assert result[sym2]["iv"] == pytest.approx(0.45, abs=1e-6)

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_batch_negative_iv_clamped(self, mock_get, _creds):
        """Batch endpoint clamps negative IV to 0.0."""
        mock_get.return_value = _mock_api_response({
            SYM: _base_snapshot(iv_key="impliedVolatility", iv_val=-1.0),
        })
        result = get_option_snapshots_batch([SYM])
        assert result[SYM]["iv"] == 0.0

    @patch(CREDS_PATCH, return_value=("https://api", "key", "secret"))
    @patch(REQ_PATCH)
    def test_batch_missing_iv_defaults_zero(self, mock_get, _creds):
        """Batch endpoint returns iv=0.0 when field is absent."""
        snap = _base_snapshot()
        snap.pop("impliedVolatility", None)
        mock_get.return_value = _mock_api_response({SYM: snap})
        result = get_option_snapshots_batch([SYM])
        assert result[SYM]["iv"] == 0.0
