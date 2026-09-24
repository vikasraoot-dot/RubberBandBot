"""
Point-in-time S&P 500 membership (source: github.com/fja05680/sp500).

The file lists the full constituent set on every date the index changed.  We use
it to (a) pick the universe from members as of 2019-12-31 and (b) compute
market breadth each day over the stocks that were members *on that day*.

Ticker renames: the membership file uses the ticker in force at the time
(e.g. FB in 2020), while Yahoo keeps the history under the current symbol
(META).  ``RENAMES`` maps well-known S&P 500 renames so the history can be
found.  Old tickers are sometimes reused by unrelated securities (Yahoo "FB"
is now a different instrument that starts in 2025); ``data_usable_for``
guards against that by requiring the price history to predate membership.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import pandas as pd

from . import config as C

# old membership ticker -> current symbol that carries the same company's history
RENAMES: Dict[str, str] = {
    "FB": "META",
    "ANTM": "ELV",
    "ABC": "COR",
    "RE": "EG",
    "PKI": "RVTY",
    "FLT": "CPAY",
    "WLTW": "WTW",
    "CTL": "LUMN",
    "NLOK": "GEN",
    "BLL": "BALL",
    "COG": "CTRA",
    "GPS": "GAP",
    "PEAK": "DOC",
    "BK": "BNY",
    "MMC": "MRSH",
    "UTX": "RTX",
    "VIAC": "PSKY",
    "MYL": "VTRS",
    "DISCA": "WBD",
}


@dataclass
class Membership:
    """Point-in-time constituent lists keyed by change date."""

    table: pd.DataFrame  # columns: date (Timestamp), members (frozenset)

    def members_on(self, date: pd.Timestamp) -> frozenset:
        """Constituents in force on ``date`` (latest change on or before it)."""
        sub = self.table[self.table["date"] <= pd.Timestamp(date)]
        if sub.empty:
            return frozenset()
        return sub.iloc[-1]["members"]

    def tickers_since(self, start: str) -> Set[str]:
        """Union of all tickers that were members at any time on/after ``start``."""
        start_ts = pd.Timestamp(start)
        out: Set[str] = set(self.members_on(start_ts))
        for m in self.table.loc[self.table["date"] > start_ts, "members"]:
            out |= set(m)
        return out

    def daily_matrix(self, sessions: pd.DatetimeIndex, tickers: List[str]) -> pd.DataFrame:
        """Boolean (session x ticker) matrix: True when the ticker was a member that session."""
        tbl = self.table.set_index("date")["members"]
        # forward-fill the membership set onto trading sessions
        idx = tbl.index.union(sessions)
        filled = tbl.reindex(idx).ffill().reindex(sessions)
        mat = pd.DataFrame(False, index=sessions, columns=tickers)
        for d, mem in filled.items():
            if isinstance(mem, frozenset):
                cols = [t for t in tickers if t in mem]
                mat.loc[d, cols] = True
        return mat


def load_membership() -> Membership:
    """Load and parse the cached fja05680 historical components file."""
    raw = pd.read_csv(C.META_DIR / "sp500_hist.csv.gz")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["members"] = raw["tickers"].map(lambda s: frozenset(x.strip() for x in s.split(",") if x.strip()))
    return Membership(raw.sort_values("date")[["date", "members"]].reset_index(drop=True))


def data_usable_for(first_data_date: pd.Timestamp, first_member_date: pd.Timestamp) -> bool:
    """
    True if a symbol's price history plausibly belongs to the index member.

    A reused ticker (history starting well after the membership began, e.g. Yahoo
    "FB" = a different instrument from 2025) is rejected.  Spin-offs that join the
    index on their first trading day (DOW, CARR, OTIS, GEHC, ...) are accepted: a
    history may start up to 5 days after the first membership date.  Histories
    starting at our download start are accepted.

    (QC fix: the original rule demanded history 60 days *before* membership, which
    wrongly excluded those spin-offs from breadth.  See REPORT.md, "Audit".)
    """
    download_start = pd.Timestamp(C.DOWNLOAD_START) + pd.Timedelta(days=7)
    return first_data_date <= max(first_member_date + pd.Timedelta(days=5), download_start)
