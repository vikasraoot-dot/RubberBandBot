"""
Cross-bot isolation on the shared RubberBand account.

1. WK_OPT exit management must only act on contracts in its own registry
   (it used to take profit / stop out 15M_OPT spread legs).
2. Daily P&L must attribute an untagged bracket exit to the bot whose registry
   held the symbol, with that entry's cost basis, even when the entry was days
   or weeks earlier (it used to require a same-day buy).
"""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from RubberBand.src.position_registry import PositionRegistry
import RubberBand.scripts.live_weekly_options_loop as wko
import RubberBand.scripts.persist_daily_results as pdr

OWN = "AAPL261016C00200000"
OTHER = "DAL261016C00050000"          # e.g. a 15M_OPT spread leg


def _pos(symbol, plpc, price="5.00"):
    return {"symbol": symbol, "unrealized_plpc": str(plpc), "current_price": price, "unrealized_pl": "0"}


def _registry(tmp_path, tag):
    return PositionRegistry(bot_tag=tag, registry_dir=str(tmp_path))


# ---------------------------------------------------------------------------- 1. WK_OPT ownership
class TestWeeklyOptionsManagesOnlyOwnContracts:

    def test_other_bots_contract_is_never_closed(self, tmp_path):
        reg = _registry(tmp_path, "WK_OPT")
        reg.record_entry(symbol=OWN, client_order_id="WK_OPT_x", qty=1, entry_price=2.0, underlying="AAPL")
        broker = [_pos(OWN, 1.50), _pos(OTHER, -0.60)]      # both meet an exit rule (TP +100%, SL -50%)
        with patch.object(wko, "get_option_positions", return_value=broker), \
             patch.object(wko, "close_option_position", return_value={"id": "ok"}) as close:
            wko.manage_weekly_positions({"tp_pct": 100.0, "sl_pct": -50.0}, MagicMock(), dry_run=False, registry=reg)
        assert [c.args[0] for c in close.call_args_list] == [OWN]
        assert OWN not in reg.positions                       # own exit recorded

    def test_no_registry_manages_nothing(self):
        with patch.object(wko, "get_option_positions", return_value=[_pos(OTHER, -0.90)]), \
             patch.object(wko, "close_option_position") as close:
            wko.manage_weekly_positions({"tp_pct": 100.0, "sl_pct": -50.0}, MagicMock(), dry_run=False, registry=None)
        close.assert_not_called()


# ---------------------------------------------------------------------------- 2. P&L attribution
def _sell(symbol, qty, price, day, oid="s1"):
    return {"id": oid, "symbol": symbol, "side": "sell", "filled_qty": str(qty), "filled_avg_price": str(price),
            "filled_at": f"{day}T15:00:00Z", "client_order_id": ""}          # untagged bracket child


def _closed(reg, symbol, qty, entry_price, entry_date, exit_date):
    reg.closed_positions.append({"symbol": symbol, "qty": qty, "entry_price": entry_price, "status": "closed",
                                 "entry_date": entry_date, "exit_date": exit_date})


class TestMultiDayAttribution:

    def test_weekly_stock_bracket_exit_uses_registry_basis(self, tmp_path):
        # Real case: BABA bought 2026-02-27 @144.03, bracket exit 2026-03-19 @121.63 (old code: -$35.88)
        wk = _registry(tmp_path, "WK_STK")
        _closed(wk, "BABA", 13, 144.03, "2026-02-27T10:00:00-05:00", "2026-03-19T16:10:00-04:00")
        regs = {"WK_STK": wk, "15M_STK": _registry(tmp_path, "15M_STK")}
        orders = [_sell("BABA", 13, 121.63, "2026-03-19")]
        r = pdr.calculate_bot_pnl(orders, [], "WK_STK", "2026-03-19", registries=regs)
        assert Decimal(r["realized_pnl"]) == Decimal("13") * (Decimal("121.63") - Decimal("144.03"))
        other = pdr.calculate_bot_pnl(orders, [], "15M_STK", "2026-03-19", registries=regs)
        assert other["trades"] == [] and Decimal(other["realized_pnl"]) == 0

    def test_open_registry_position_also_owns_the_exit(self, tmp_path):
        wk = _registry(tmp_path, "WK_STK")
        wk.positions["QCOM"] = {"symbol": "QCOM", "qty": 5, "entry_price": 150.0, "status": "open",
                                "entry_date": "2026-09-01T10:00:00-04:00"}
        r = pdr.calculate_bot_pnl([_sell("QCOM", 5, 160, "2026-09-18")], [], "WK_STK", "2026-09-18",
                                  registries={"WK_STK": wk})
        assert Decimal(r["realized_pnl"]) == Decimal("50")

    def test_symbol_held_by_two_bots_is_not_attributed(self, tmp_path):
        a, b = _registry(tmp_path, "WK_STK"), _registry(tmp_path, "15M_STK")
        for reg in (a, b):
            _closed(reg, "NVDA", 2, 100.0, "2026-09-10T10:00:00-04:00", "2026-09-18T12:00:00-04:00")
        orders = [_sell("NVDA", 2, 110, "2026-09-18")]
        for tag in ("WK_STK", "15M_STK"):
            r = pdr.calculate_bot_pnl(orders, [], tag, "2026-09-18", registries={"WK_STK": a, "15M_STK": b})
            assert r["trades"] == []

    def test_stale_closed_record_does_not_claim_a_later_sell(self, tmp_path):
        wk = _registry(tmp_path, "WK_STK")
        _closed(wk, "AMD", 3, 90.0, "2026-08-01T10:00:00-04:00", "2026-08-15T12:00:00-04:00")
        r = pdr.calculate_bot_pnl([_sell("AMD", 3, 120, "2026-09-18")], [], "WK_STK", "2026-09-18",
                                  registries={"WK_STK": wk})
        assert r["trades"] == []

    def test_option_exit_uses_contract_multiplier(self, tmp_path):
        wk = _registry(tmp_path, "WK_OPT")
        _closed(wk, OWN, 1, 16.17, "2026-08-20T10:00:00-04:00", "2026-09-18T11:00:00-04:00")
        r = pdr.calculate_bot_pnl([_sell(OWN, 1, 20.00, "2026-09-18")], [], "WK_OPT", "2026-09-18",
                                  registries={"WK_OPT": wk})
        assert Decimal(r["realized_pnl"]) == (Decimal("20.00") - Decimal("16.17")) * 100

    def test_owned_exit_without_basis_is_flagged_not_zeroed_silently(self, tmp_path):
        wk = _registry(tmp_path, "WK_STK")
        _closed(wk, "EQT", 4, 0.0, "2026-09-01T10:00:00-04:00", "2026-09-18T12:00:00-04:00")
        r = pdr.calculate_bot_pnl([_sell("EQT", 4, 50, "2026-09-18")], [], "WK_STK", "2026-09-18",
                                  registries={"WK_STK": wk})
        assert r["unpriced_exits"] == [{"symbol": "EQT", "qty": "4"}] and Decimal(r["realized_pnl"]) == 0

    def test_without_registries_behaviour_is_unchanged(self):
        # Legacy call signature: untagged sell with no same-day buy stays unattributed
        r = pdr.calculate_bot_pnl([_sell("BABA", 13, 121.63, "2026-03-19")], [], "WK_STK", "2026-03-19")
        assert r["trades"] == []


class TestReviewFollowUps:

    def test_same_day_buyer_and_registry_holder_is_ambiguous(self, tmp_path):
        # 15M_STK buys QCOM today while WK_STK's registry already holds QCOM:
        # the untagged bracket sell must not be credited to both (or either).
        wk = _registry(tmp_path, "WK_STK")
        wk.positions["QCOM"] = {"symbol": "QCOM", "qty": 10, "entry_price": 165.255, "status": "open",
                                "entry_date": "2026-09-10T10:00:00-04:00"}
        buy = {"id": "b1", "symbol": "QCOM", "side": "buy", "filled_qty": "10", "filled_avg_price": "150",
               "filled_at": "2026-09-18T14:00:00Z", "client_order_id": "15M_STK_QCOM_1"}
        orders = [buy, _sell("QCOM", 10, 152, "2026-09-18")]
        regs = {"WK_STK": wk, "15M_STK": _registry(tmp_path, "15M_STK")}
        with patch.object(pdr, "extract_bot_tag_from_order", side_effect=lambda o: "15M_STK" if o["id"] == "b1" else None):
            wk_r = pdr.calculate_bot_pnl(orders, [], "WK_STK", "2026-09-18", registries=regs)
            st_r = pdr.calculate_bot_pnl(orders, [], "15M_STK", "2026-09-18", registries=regs)
        assert wk_r["trades"] == [] and Decimal(wk_r["realized_pnl"]) == 0
        assert [t["side"] for t in st_r["trades"]] == ["buy"] and Decimal(st_r["realized_pnl"]) == 0

    def test_closed_record_without_exit_date_is_ignored(self, tmp_path):
        wk = _registry(tmp_path, "WK_STK")
        wk.closed_positions.append({"symbol": "PDD", "qty": 3, "entry_price": 100.0, "status": "closed",
                                    "entry_date": "2026-08-01T10:00:00-04:00"})
        r = pdr.calculate_bot_pnl([_sell("PDD", 3, 90, "2026-09-18")], [], "WK_STK", "2026-09-18",
                                  registries={"WK_STK": wk})
        assert r["trades"] == []

    def test_registry_basis_prices_at_most_the_record_qty(self, tmp_path):
        wk = _registry(tmp_path, "WK_STK")
        _closed(wk, "VRT", 5, 100.0, "2026-09-01T10:00:00-04:00", "2026-09-18T12:00:00-04:00")
        r = pdr.calculate_bot_pnl([_sell("VRT", 8, 110, "2026-09-18")], [], "WK_STK", "2026-09-18",
                                  registries={"WK_STK": wk})
        assert Decimal(r["realized_pnl"]) == Decimal("50")
        assert r["unpriced_exits"] == [{"symbol": "VRT", "qty": "3"}]

    def test_registries_are_loaded_before_reconcile_deletes_orphans(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(pdr, "DAILY_RESULTS_DIR", str(tmp_path / "daily"))
        monkeypatch.setattr(pdr, "ensure_all_registries_exist", lambda *a, **k: [])
        monkeypatch.setattr(pdr, "PositionRegistry", lambda bot_tag, **k: calls.append(("load", bot_tag)) or MagicMock(
            positions={}, closed_positions=[]))
        monkeypatch.setattr(pdr, "reconcile_positions", lambda **k: calls.append(("reconcile",)) or {})
        monkeypatch.setattr(pdr, "get_positions", lambda *a, **k: [])
        monkeypatch.setattr(pdr, "get_orders_for_week", lambda *a, **k: [])
        monkeypatch.setattr(pdr, "get_account_info", lambda *a, **k: {})
        pdr.persist_daily_results("2026-09-18")
        first_reconcile = calls.index(("reconcile",))
        assert first_reconcile > 0 and all(c[0] == "load" for c in calls[:first_reconcile])
