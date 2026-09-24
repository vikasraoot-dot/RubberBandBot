"""
Step 2: construct and freeze the 100-stock universe.

Rules (all point-in-time as of 2019-12-31, i.e. *before* the backtest starts):
1. Candidate pool = S&P 500 members on 2019-12-31 (fja05680 point-in-time list).
2. 2019 metrics from daily bars: median as-traded close, median dollar volume,
   median ATR14/close, realised volatility.
3. Eligibility: median 2019 price > $10, median 2019 dollar volume > $100M,
   usable price history covering 2019 (see ``membership.data_usable_for``).
4. Sector quotas (``config.SECTOR_BUCKET_TARGETS``) using GICS sector /
   sub-industry; within each bucket take the most liquid names by 2019 median
   dollar volume.  Duplicate share classes are collapsed.
No strategy output is used anywhere in this selection.

Known limitation: members that were later acquired/delisted (e.g. ATVI, TWTR,
XLNX) have no Yahoo history, so they cannot be selected.  They are listed in
``data/meta/universe_excluded_no_data.csv`` for transparency.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .features import wilder_atr
from .membership import RENAMES, data_usable_for, load_membership

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("universe")

SEMI_SUBS = {"Semiconductors", "Semiconductor Materials & Equipment", "Semiconductor Equipment"}
SOFTWARE_SUBS = {"Application Software", "Systems Software"}
TRANSPORT_SUBS = {"Air Freight & Logistics", "Passenger Airlines", "Airlines", "Rail Transportation",
                  "Railroads", "Cargo Ground Transportation", "Trucking", "Marine Transportation",
                  "Passenger Ground Transportation", "Marine"}
SHARE_CLASS_GROUPS = [["GOOGL", "GOOG"], ["FOXA", "FOX"], ["NWSA", "NWS"], ["DISCA", "DISCK"], ["UAA", "UA"]]

# GICS classification for 2019 members that are no longer in the current Wikipedia table
# (removed from the index but still trading).  Only needed for names with usable data.
MANUAL_GICS: Dict[str, tuple] = {
    "AAL": ("American Airlines Group", "Industrials", "Passenger Airlines"),
    "AAP": ("Advance Auto Parts", "Consumer Discretionary", "Automotive Retail"),
    "AIV": ("Apartment Investment & Management", "Real Estate", "Multi-Family Residential REITs"),
    "ALK": ("Alaska Air Group", "Industrials", "Passenger Airlines"),
    "BWA": ("BorgWarner", "Consumer Discretionary", "Automotive Parts & Equipment"),
    "CAG": ("Conagra Brands", "Consumer Staples", "Packaged Foods & Meats"),
    "CE": ("Celanese", "Materials", "Specialty Chemicals"),
    "COTY": ("Coty", "Consumer Staples", "Personal Care Products"),
    "CPB": ("Campbell's", "Consumer Staples", "Packaged Foods & Meats"),
    "CPRI": ("Capri Holdings", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "CTL": ("CenturyLink / Lumen", "Communication Services", "Integrated Telecommunication Services"),
    "DXC": ("DXC Technology", "Information Technology", "IT Consulting & Other Services"),
    "EMN": ("Eastman Chemical", "Materials", "Specialty Chemicals"),
    "FLS": ("Flowserve", "Industrials", "Industrial Machinery & Supplies & Components"),
    "FMC": ("FMC Corp", "Materials", "Fertilizers & Agricultural Chemicals"),
    "FTI": ("TechnipFMC", "Energy", "Oil & Gas Equipment & Services"),
    "GPS": ("Gap Inc", "Consumer Discretionary", "Apparel Retail"),
    "HOG": ("Harley-Davidson", "Consumer Discretionary", "Motorcycle Manufacturers"),
    "HP": ("Helmerich & Payne", "Energy", "Oil & Gas Drilling"),
    "HRB": ("H&R Block", "Consumer Discretionary", "Specialized Consumer Services"),
    "IPGP": ("IPG Photonics", "Information Technology", "Electronic Manufacturing Services"),
    "KMX": ("CarMax", "Consumer Discretionary", "Automotive Retail"),
    "KSS": ("Kohl's", "Consumer Discretionary", "Broadline Retail"),
    "LKQ": ("LKQ Corp", "Consumer Discretionary", "Distributors"),
    "LNC": ("Lincoln National", "Financials", "Life & Health Insurance"),
    "LW": ("Lamb Weston", "Consumer Staples", "Packaged Foods & Meats"),
    "M": ("Macy's", "Consumer Discretionary", "Broadline Retail"),
    "MHK": ("Mohawk Industries", "Consumer Discretionary", "Home Furnishings"),
    "MKTX": ("MarketAxess", "Financials", "Financial Exchanges & Data"),
    "NOV": ("NOV Inc", "Energy", "Oil & Gas Equipment & Services"),
    "NWL": ("Newell Brands", "Consumer Discretionary", "Housewares & Specialties"),
    "PRGO": ("Perrigo", "Health Care", "Pharmaceuticals"),
    "PVH": ("PVH Corp", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "QRVO": ("Qorvo", "Information Technology", "Semiconductors"),
    "RHI": ("Robert Half", "Industrials", "Human Resource & Employment Services"),
    "SLG": ("SL Green Realty", "Real Estate", "Office REITs"),
    "TAP": ("Molson Coors", "Consumer Staples", "Brewers"),
    "TFX": ("Teleflex", "Health Care", "Health Care Equipment"),
    "UA": ("Under Armour (Class C)", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "UAA": ("Under Armour (Class A)", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "UNM": ("Unum Group", "Financials", "Life & Health Insurance"),
    "VFC": ("VF Corp", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "VNO": ("Vornado Realty Trust", "Real Estate", "Office REITs"),
    "WHR": ("Whirlpool", "Consumer Discretionary", "Household Appliances"),
    "WU": ("Western Union", "Financials", "Transaction & Payment Processing Services"),
    "XRAY": ("Dentsply Sirona", "Health Care", "Health Care Supplies"),
    "XRX": ("Xerox", "Information Technology", "Technology Hardware, Storage & Peripherals"),
    "ZION": ("Zions Bancorporation", "Financials", "Regional Banks"),
}


def bucket_for(sector: str, sub: str) -> Optional[str]:
    """Map a GICS sector / sub-industry to one of the study's sector buckets."""
    if sector == "Information Technology":
        if sub in SEMI_SUBS:
            return "Semiconductors"
        if sub in SOFTWARE_SUBS:
            return "Software"
        return "Tech Hardware & IT Services"
    if sector == "Communication Services":
        return "Communication/Internet"
    if sector == "Industrials":
        return "Transportation" if sub in TRANSPORT_SUBS else "Industrials"
    if sector == "Health Care":
        return "Biotech" if sub == "Biotechnology" else "Healthcare"
    return {"Consumer Discretionary": "Consumer Discretionary", "Consumer Staples": "Consumer Staples",
            "Financials": "Financials", "Energy": "Energy", "Materials": "Materials",
            "Utilities": "Utilities", "Real Estate": "Real Estate"}.get(sector)


def load_gics() -> pd.DataFrame:
    """Current GICS sector/sub-industry per ticker from the cached Wikipedia page."""
    import gzip
    from io import StringIO
    with gzip.open(C.META_DIR / "sp500_wikipedia.html.gz", "rt", encoding="utf-8") as fh:
        t = pd.read_html(StringIO(fh.read()))[0]
    t = t.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector",
                          "GICS Sub-Industry": "industry"})
    return t[["ticker", "name", "sector", "industry"]].set_index("ticker")


def resolve_symbol(ticker: str, first_member: pd.Timestamp) -> Optional[str]:
    """Return the cached symbol whose history plausibly belongs to ``ticker``, else None."""
    for sym in (RENAMES.get(ticker), ticker):
        if sym is None:
            continue
        try:
            df = D.load_cached(sym, C.BREADTH_RAW_DIR)
        except FileNotFoundError:
            continue
        if len(df) and data_usable_for(df.index.min(), first_member):
            return sym
    return None


def metrics_2019(df: pd.DataFrame) -> Dict[str, float]:
    """Point-in-time liquidity / volatility metrics over calendar 2019."""
    w = df.loc[C.UNIVERSE_SELECTION_START:C.UNIVERSE_SELECTION_END]
    atr = wilder_atr(df["high"], df["low"], df["close"], C.ATR_LEN).loc[w.index]
    split_close = w["close"] / w["adj_factor"]
    dollar_vol = w["volume"] * split_close
    lr = np.log(w["close"]).diff().dropna()
    return {
        "price_2019_median": float(w["raw_close"].median()),
        "median_dollar_volume_2019": float(dollar_vol.median()),
        "atr_pct_2019": float((atr / w["close"]).median()),
        "realized_vol_2019": float(lr.std() * np.sqrt(252)),
        "sessions_2019": int(len(w)),
    }


def build() -> pd.DataFrame:
    """Construct the universe and write ``universe_100.csv`` plus diagnostics."""
    mem = load_membership()
    members = sorted(mem.members_on(pd.Timestamp(C.UNIVERSE_MEMBERSHIP_DATE)))
    gics = load_gics()
    first_member = pd.Timestamp(C.UNIVERSE_MEMBERSHIP_DATE)
    rows, no_data, no_gics = [], [], []
    for t in members:
        sym = resolve_symbol(t, first_member - pd.Timedelta(days=365))
        if sym is None:
            no_data.append(t)
            continue
        info = None
        if t in MANUAL_GICS:
            info = pd.Series(dict(zip(["name", "sector", "industry"], MANUAL_GICS[t])))
        for key in (sym, t) if info is None else ():
            if key in gics.index:
                info = gics.loc[key]
                break
        if info is None:
            no_gics.append(t)
            continue
        df = D.load_cached(sym, C.BREADTH_RAW_DIR)
        m = metrics_2019(df)
        full = df.loc[C.BACKTEST_START:]
        rows.append({
            "ticker": sym, "member_ticker_2019": t, "name": info["name"], "sector": info["sector"],
            "industry": info["industry"], "bucket": bucket_for(info["sector"], info["industry"]),
            **m,
            "price_latest": float(df["raw_close"].iloc[-1]),
            "history_start": df.index.min().date().isoformat(),
            "history_end": df.index.max().date().isoformat(),
            "atr_pct_full_period": float((wilder_atr(df["high"], df["low"], df["close"], C.ATR_LEN)
                                          / df["close"]).loc[C.BACKTEST_START:].median()),
            "realized_vol_full_period": float(np.log(full["close"]).diff().std() * np.sqrt(252)),
        })
    cand = pd.DataFrame(rows)
    # collapse duplicate share classes: keep the most liquid class
    for grp in SHARE_CLASS_GROUPS:
        present = cand[cand["ticker"].isin(grp)]
        if len(present) > 1:
            keep = present["median_dollar_volume_2019"].idxmax()
            cand = cand.drop(present.index.difference([keep]))
    cand["eligible"] = ((cand["price_2019_median"] > C.UNIVERSE_MIN_PRICE)
                        & (cand["median_dollar_volume_2019"] > C.UNIVERSE_MIN_MEDIAN_DOLLAR_VOLUME)
                        & (cand["sessions_2019"] >= 240))
    picks: List[pd.DataFrame] = []
    for bucket, n in C.SECTOR_BUCKET_TARGETS.items():
        pool = cand[cand["bucket"] == bucket].sort_values("median_dollar_volume_2019", ascending=False)
        strict = pool[pool["eligible"]].head(n).copy()
        strict["inclusion_reason"] = (f"top {n} {bucket} by 2019 median $ volume; "
                                      f"price>${C.UNIVERSE_MIN_PRICE:.0f}, $vol>${C.UNIVERSE_MIN_MEDIAN_DOLLAR_VOLUME/1e6:.0f}M")
        if len(strict) < n:
            relaxed_pool = pool[(pool["price_2019_median"] > C.UNIVERSE_MIN_PRICE)
                                & (pool["sessions_2019"] >= 240) & ~pool.index.isin(strict.index)]
            extra = relaxed_pool.head(n - len(strict)).copy()
            extra["inclusion_reason"] = (f"{bucket} quota not met at $100M; next most liquid by 2019 "
                                         f"median $ volume (liquidity threshold relaxed)")
            strict = pd.concat([strict, extra])
        picks.append(strict)
    uni = pd.concat(picks).reset_index(drop=True)

    # volatility buckets (quartiles of point-in-time 2019 ATR%)
    q = uni["atr_pct_2019"].quantile([0.25, 0.5, 0.75]).to_numpy()
    uni["vol_bucket"] = np.select([uni["atr_pct_2019"] <= q[0], uni["atr_pct_2019"] <= q[1],
                                   uni["atr_pct_2019"] <= q[2]], ["Low", "Medium", "High"], default="Very High")
    uni = uni.sort_values(["bucket", "median_dollar_volume_2019"], ascending=[True, False])
    cols = ["ticker", "member_ticker_2019", "name", "sector", "industry", "bucket", "price_2019_median",
            "price_latest", "median_dollar_volume_2019", "atr_pct_2019", "realized_vol_2019", "vol_bucket",
            "atr_pct_full_period", "realized_vol_full_period", "history_start", "history_end", "inclusion_reason"]
    uni[cols].to_csv(C.OUT_DIR / "universe_100.csv", index=False)

    # diagnostics: what the selection could not see
    excl = pd.DataFrame({"member_ticker_2019": no_data, "reason": "no usable Yahoo history (acquired/delisted/renamed)"})
    excl.to_csv(C.META_DIR / "universe_excluded_no_data.csv", index=False)
    cand.to_csv(C.META_DIR / "universe_candidates_2019.csv", index=False)
    log.info("members=%d usable=%d no_data=%d no_gics=%s universe=%d",
             len(members), len(cand), len(no_data), no_gics, len(uni))
    return uni


if __name__ == "__main__":
    build()
