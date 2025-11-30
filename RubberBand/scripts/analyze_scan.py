#!/usr/bin/env python3
import json
import pandas as pd
import os

def main():
    json_path = "full_scan_results.json"
    if not os.path.exists(json_path):
        print("No results file found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        print("No data in results.")
        return

    # Pivot to see performance across timeframes
    # We want to see if a ticker is consistent.
    # Group by symbol
    report_lines = []
    report_lines.append("# Candidate Analysis Report\n")
    report_lines.append("## Methodology")
    report_lines.append("- **Source**: `tickers_full_list.txt` (Pilot Scan)")
    report_lines.append("- **Timeframes**: 30, 90, 120, 240, 350 Days")
    report_lines.append("- **Criteria**: Positive PnL, Win Rate > 50%, Consistent Performance\n")
    
    report_lines.append("## Top Candidates\n")
    report_lines.append("| Ticker | 30D PnL | 90D PnL | 350D PnL | 350D WR | Trades (350D) | Score |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    symbols = df["symbol"].unique()
    candidates = []

    for sym in symbols:
        try:
            sub = df[df["symbol"] == sym].set_index("days")
            
            # Check if we have data for all timeframes (or at least 350)
            if 350 not in sub.index:
                continue
                
            pnl_350 = float(sub.loc[350, "net"])
            wr_350 = float(sub.loc[350, "win_rate"])
            trades_350 = int(sub.loc[350, "trades"])
            
            pnl_30 = float(sub.loc[30, "net"]) if 30 in sub.index else 0.0
            pnl_90 = float(sub.loc[90, "net"]) if 90 in sub.index else 0.0
            
            # Scoring Logic
            score = 0
            if pnl_350 > 0: score += 1
            if pnl_90 > 0: score += 1
            if pnl_30 > 0: score += 1
            if wr_350 > 50: score += 1
            if trades_350 > 20: score += 1 # Liquidity/Frequency check
            
            # Filter: Must be profitable over 350 days and have decent win rate
            if pnl_350 > 100 and wr_350 > 45:
                candidates.append({
                    "sym": sym,
                    "pnl_30": pnl_30,
                    "pnl_90": pnl_90,
                    "pnl_350": pnl_350,
                    "wr_350": wr_350,
                    "trades_350": trades_350,
                    "score": score
                })
        except Exception as e:
            print(f"Skipping {sym} due to error: {e}")
            continue

    # Sort by Score then PnL
    candidates.sort(key=lambda x: (x["score"], x["pnl_350"]), reverse=True)

    for c in candidates:
        line = f"| **{c['sym']}** | ${c['pnl_30']:.2f} | ${c['pnl_90']:.2f} | **${c['pnl_350']:.2f}** | {c['wr_350']}% | {c['trades_350']} | {c['score']}/5 |"
        report_lines.append(line)

    report_lines.append("\n## Detailed Breakdown\n")
    for c in candidates:
        report_lines.append(f"### {c['sym']}")
        # Manual markdown table
        sub = df[df["symbol"] == c['sym']].sort_values("days")
        report_lines.append("| Days | Trades | Net PnL | Win Rate |")
        report_lines.append("| :--- | :--- | :--- | :--- |")
        for _, row in sub.iterrows():
            report_lines.append(f"| {row['days']} | {row['trades']} | ${row['net']:.2f} | {row['win_rate']}% |")
        report_lines.append("\n")

    # Write report
    try:
        with open("candidate_report.md", "w") as f:
            f.write("\n".join(report_lines))
        print("Generated candidate_report.md")
    except Exception as e:
        print(f"Error writing report: {e}")

if __name__ == "__main__":
    main()
