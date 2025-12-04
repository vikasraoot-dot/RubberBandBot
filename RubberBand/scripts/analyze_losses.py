#!/usr/bin/env python3
"""
Analyze losing trades to identify patterns and recommend screening improvements.
"""
import pandas as pd
import argparse

def analyze_losses(loss_file="loss_analysis.csv", all_trades_file="detailed_trades.csv"):
    """Analyze loss patterns and generate recommendations."""
    
    import os
    
    # Check file existence
    if not os.path.exists(all_trades_file):
        print(f"ERROR: {all_trades_file} not found. Run backtest first.")
        return
    
    # Load data
    df_all = pd.read_csv(all_trades_file)
    
    if df_all.empty:
        print("No trades to analyze. Run backtest with more tickers or longer period.")
        return
    
    # Check if loss file exists (might not exist if all trades were profitable)
    if os.path.exists(loss_file):
        df_losses = pd.read_csv(loss_file)
    else:
        print(f"No {loss_file} found. Filtering losses from all trades.")
        df_losses = df_all[df_all["pnl"] <= 0].copy()
        if df_losses.empty:
            print("✅ No losing trades! All trades were profitable.")
            return
    
    
    # Clean data - drop rows with NaN in critical columns
    df_all_clean = df_all.dropna(subset=["pnl", "entry_rsi", "entry_atr", "entry_price", "hold_duration_days"])
    
    if df_all_clean.empty:
        print("WARNING: All trades have missing data. Cannot perform detailed analysis.")
        # Still show basic stats
        df_all_clean = df_all  # Use original for basic stats
    
    total_trades = len(df_all)
    total_losses = len(df_losses)
    loss_rate = (total_losses / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = df_all["pnl"].sum()
    total_loss_pnl = df_losses["pnl"].sum()
    
    print(f"\n{'='*60}")
    print(f"LOSS ANALYSIS REPORT")
    print(f"{'='*60}\n")
    
    print(f"Total Trades: {total_trades}")
    print(f"Losing Trades: {total_losses} ({loss_rate:.1f}%)")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Total Loss PnL: ${total_loss_pnl:.2f}")
    if total_losses > 0:
        print(f"Average Loss: ${df_losses['pnl'].mean():.2f}\n")
    else:
        print(f"Average Loss: N/A\n")
    
    # 1. Analysis by Exit Reason
    print(f"\n{'='*60}")
    print("1. LOSSES BY EXIT REASON")
    print(f"{'='*60}\n")
    
    exit_reason_stats = df_losses.groupby("exit_reason").agg({
        "pnl": ["count", "sum", "mean"]
    }).round(2)
    exit_reason_stats.columns = ["Count", "Total Loss $", "Avg Loss $"]
    print(exit_reason_stats.sort_values("Total Loss $"))
    
    # 2. Analysis by Ticker
    print(f"\n{'='*60}")
    print("2. WORST PERFORMING TICKERS (Top 20)")
    print(f"{'='*60}\n")
    
    ticker_stats = df_all.groupby("symbol").agg({
        "pnl": ["count", "sum", lambda x: (x > 0).sum() / len(x) * 100]
    }).round(2)
    ticker_stats.columns = ["Trades", "Net PnL $", "Win Rate %"]
    ticker_stats = ticker_stats[ticker_stats["Trades"] >= 3]  # Min 3 trades
    worst_tickers = ticker_stats.sort_values("Net PnL $").head(20)
    print(worst_tickers)
    
    # 3. Analysis by Entry Conditions
    print(f"\n{'='*60}")
    print("3. ENTRY CONDITION ANALYSIS")
    print(f"{'='*60}\n")
    
    # RSI Analysis
    df_all_clean["rsi_bucket"] = pd.cut(df_all_clean["entry_rsi"], bins=[0, 20, 25, 30, 35, 100], 
                                   labels=["<20", "20-25", "25-30", "30-35", ">35"])
    rsi_analysis = df_all_clean.groupby("rsi_bucket").agg({
        "pnl": ["count", lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0, "mean"]
    }).round(2)
    rsi_analysis.columns = ["Trades", "Win Rate %", "Avg PnL $"]
    print("Entry RSI Distribution:")
    print(rsi_analysis)
    
    # ATR Analysis
    df_all_clean["atr_pct"] = (df_all_clean["entry_atr"] / df_all_clean["entry_price"] * 100)
    df_all_clean["atr_bucket"] = pd.cut(df_all_clean["atr_pct"], bins=[0, 1.5, 2.5, 4, 100], 
                                   labels=["<1.5%", "1.5-2.5%", "2.5-4%", ">4%"])
    atr_analysis = df_all_clean.groupby("atr_bucket").agg({
        "pnl": ["count", lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0, "mean"]
    }).round(2)
    atr_analysis.columns = ["Trades", "Win Rate %", "Avg PnL $"]
    print("\nEntry ATR% Distribution:")
    print(atr_analysis)
    
    # 4. Analysis by Hold Duration
    print(f"\n{'='*60}")
    print("4. HOLD DURATION ANALYSIS")
    print(f"{'='*60}\n")
    
    df_all_clean["hold_bucket"] = pd.cut(df_all_clean["hold_duration_days"], 
                                    bins=[0, 0.1, 0.5, 1, 100], 
                                    labels=["<2.4h", "2.4h-12h", "12h-1d", ">1d"])
    hold_analysis = df_all_clean.groupby("hold_bucket").agg({
        "pnl": ["count", lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0, "mean"]
    }).round(2)
    hold_analysis.columns = ["Trades", "Win Rate %", "Avg PnL $"]
    print(hold_analysis)
    
    # 5. Recommendations
    print(f"\n{'='*60}")
    print("5. RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    recommendations = []
    
    # Check if SL exits are dominant
    sl_losses = df_losses[df_losses["exit_reason"] == "SL"]
    if len(df_losses) > 0 and len(sl_losses) / len(df_losses) > 0.4:
        recommendations.append(
            f"⚠️  {len(sl_losses)/len(df_losses)*100:.1f}% of losses are from Stop Loss hits. "
            "Consider tightening entry criteria or widening SL."
        )
    
    # Check for bad tickers
    bad_tickers = worst_tickers[worst_tickers["Win Rate %"] < 40].index.tolist()
    if bad_tickers:
        recommendations.append(
            f"🚫 Exclude {len(bad_tickers)} tickers with <40% win rate: {', '.join(bad_tickers[:10])}"
        )
    
    # Check RSI threshold
    high_rsi_losses = df_losses[df_losses["entry_rsi"] > 30]
    if len(df_losses) > 0 and len(high_rsi_losses) / len(df_losses) > 0.3:
        recommendations.append(
            f"📊 {len(high_rsi_losses)/len(df_losses)*100:.1f}% of losses had entry RSI > 30. "
            "Consider lowering RSI threshold to <28."
        )
    
    # Check ATR
    low_atr_trades = df_all_clean[df_all_clean["atr_pct"] < 1.5]
    if len(low_atr_trades) > 0:
        low_atr_wr = (low_atr_trades["pnl"] > 0).sum() / len(low_atr_trades) * 100
        if low_atr_wr < 55:
            recommendations.append(
                f"📉 Trades with ATR% < 1.5% have {low_atr_wr:.1f}% win rate. "
                "Consider increasing min ATR% to 2.0% in scanner."
            )
    
    # Check EOD flattening impact
    eod_exits = df_all_clean[df_all_clean["exit_reason"] == "EOD"]
    if len(eod_exits) > 0:
        eod_wr = (eod_exits["pnl"] > 0).sum() / len(eod_exits) * 100 if len(eod_exits) > 0 else 0
        eod_avg_pnl = eod_exits["pnl"].mean()
        recommendations.append(
            f"🌙 EOD Flattening: {len(eod_exits)} trades ({eod_wr:.1f}% WR, ${eod_avg_pnl:.2f} avg PnL). "
            f"{'Consider disabling if profitable' if eod_avg_pnl > 0 else 'Effective at cutting losses'}"
        )
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}\n")
    else:
        print("✅ No major issues detected. Strategy is performing well!")
    
    # Save summary
    with open("loss_patterns_report.txt", "w") as f:
        f.write(f"LOSS ANALYSIS REPORT\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total Trades: {total_trades}\n")
        f.write(f"Losing Trades: {total_losses} ({loss_rate:.1f}%)\n")
        f.write(f"Total Loss PnL: ${total_loss_pnl:.2f}\n\n")
        f.write("RECOMMENDATIONS:\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
    
    print(f"\n📄 Full report saved to loss_patterns_report.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze losing trades")
    parser.add_argument("--losses", default="loss_analysis.csv", help="Loss analysis CSV file")
    parser.add_argument("--all-trades", default="detailed_trades.csv", help="All trades CSV file")
    args = parser.parse_args()
    
    analyze_losses(args.losses, args.all_trades)
