#!/usr/bin/env python3
"""
Deep dive analysis of losing trades to identify actionable patterns.
"""
import pandas as pd
import numpy as np

# Load data (using WITH EOD flattening results for consistency)
print("Loading trade data...")
print("NOTE: Using results WITH EOD flattening (the better performing config)")
print()

# We need to re-run the backtest with EOD to get the loss_analysis.csv
# For now, let's create a comprehensive analysis script

df_all = pd.read_csv('detailed_trades.csv')
df_losses = df_all[df_all['pnl'] <= 0].copy()

print(f"{'='*80}")
print(f"DEEP DIVE: LOSING TRADES ANALYSIS")
print(f"{'='*80}\n")

print(f"Total Trades: {len(df_all)}")
print(f"Losing Trades: {len(df_losses)} ({len(df_losses)/len(df_all)*100:.1f}%)")
print(f"Total Loss: ${df_losses['pnl'].sum():.2f}")
print(f"Average Loss: ${df_losses['pnl'].mean():.2f}\n")

# 1. TICKER ANALYSIS - Which tickers are chronic losers?
print(f"\n{'='*80}")
print("1. WORST PERFORMING TICKERS")
print(f"{'='*80}\n")

ticker_stats = df_all.groupby('symbol').agg({
    'pnl': ['count', 'sum', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0]
}).round(2)
ticker_stats.columns = ['Trades', 'Net_PnL', 'Win_Rate']
ticker_stats = ticker_stats[ticker_stats['Trades'] >= 3]  # Min 3 trades
worst_tickers = ticker_stats.sort_values('Net_PnL').head(30)

print("Top 30 Worst Tickers (Min 3 trades):")
print(worst_tickers)

# Identify tickers to exclude
bad_tickers = worst_tickers[worst_tickers['Win_Rate'] < 40].index.tolist()
print(f"\n🚫 RECOMMENDATION: Exclude {len(bad_tickers)} tickers with <40% win rate:")
print(f"   {', '.join(bad_tickers[:20])}")
if len(bad_tickers) > 20:
    print(f"   ... and {len(bad_tickers)-20} more")

# 2. ENTRY CONDITION ANALYSIS
print(f"\n{'='*80}")
print("2. ENTRY CONDITION PATTERNS IN LOSSES")
print(f"{'='*80}\n")

# RSI Analysis
print("A. Entry RSI Analysis:")
print(f"   Losses - Avg RSI: {df_losses['entry_rsi'].mean():.2f}")
print(f"   Winners - Avg RSI: {df_all[df_all['pnl'] > 0]['entry_rsi'].mean():.2f}")

rsi_buckets = pd.cut(df_all['entry_rsi'], bins=[0, 20, 25, 28, 30, 35, 100])
rsi_analysis = df_all.groupby(rsi_buckets).agg({
    'pnl': ['count', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0, 'mean']
}).round(2)
rsi_analysis.columns = ['Trades', 'Win_Rate_%', 'Avg_PnL']
print("\n   RSI Bucket Performance:")
print(rsi_analysis)

# Find optimal RSI threshold
high_rsi_losses = df_losses[df_losses['entry_rsi'] > 28]
print(f"\n   Losses with RSI > 28: {len(high_rsi_losses)} ({len(high_rsi_losses)/len(df_losses)*100:.1f}%)")

# ATR% Analysis
print("\nB. Entry ATR% Analysis:")
df_all['atr_pct'] = (df_all['entry_atr'] / df_all['entry_price'] * 100)
df_losses['atr_pct'] = (df_losses['entry_atr'] / df_losses['entry_price'] * 100)

print(f"   Losses - Avg ATR%: {df_losses['atr_pct'].mean():.2f}%")
print(f"   Winners - Avg ATR%: {df_all[df_all['pnl'] > 0]['atr_pct'].mean():.2f}%")

atr_buckets = pd.cut(df_all['atr_pct'], bins=[0, 1.5, 2.0, 2.5, 3.0, 4.0, 100])
atr_analysis = df_all.groupby(atr_buckets).agg({
    'pnl': ['count', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0, 'mean']
}).round(2)
atr_analysis.columns = ['Trades', 'Win_Rate_%', 'Avg_PnL']
print("\n   ATR% Bucket Performance:")
print(atr_analysis)

# 3. EXIT REASON ANALYSIS
print(f"\n{'='*80}")
print("3. EXIT REASON BREAKDOWN FOR LOSSES")
print(f"{'='*80}\n")

loss_exit_reasons = df_losses.groupby('exit_reason').agg({
    'pnl': ['count', 'sum', 'mean']
}).round(2)
loss_exit_reasons.columns = ['Count', 'Total_Loss', 'Avg_Loss']
print(loss_exit_reasons.sort_values('Total_Loss'))

# 4. HOLD DURATION ANALYSIS
print(f"\n{'='*80}")
print("4. HOLD DURATION PATTERNS")
print(f"{'='*80}\n")

print(f"Losses - Avg Hold: {df_losses['hold_duration_days'].mean():.2f} days")
print(f"Winners - Avg Hold: {df_all[df_all['pnl'] > 0]['hold_duration_days'].mean():.2f} days")

hold_buckets = pd.cut(df_all['hold_duration_days'], bins=[0, 0.02, 0.1, 0.5, 1.0, 100])
hold_analysis = df_all.groupby(hold_buckets).agg({
    'pnl': ['count', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0, 'mean']
}).round(2)
hold_analysis.columns = ['Trades', 'Win_Rate_%', 'Avg_PnL']
print("\nHold Duration Performance:")
print(hold_analysis)

# 5. TIME OF DAY ANALYSIS
print(f"\n{'='*80}")
print("5. TIME OF DAY ANALYSIS")
print(f"{'='*80}\n")

df_all['entry_hour'] = pd.to_datetime(df_all['entry_time']).dt.hour
df_losses['entry_hour'] = pd.to_datetime(df_losses['entry_time']).dt.hour

hour_analysis = df_all.groupby('entry_hour').agg({
    'pnl': ['count', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0, 'mean']
}).round(2)
hour_analysis.columns = ['Trades', 'Win_Rate_%', 'Avg_PnL']
print("Performance by Entry Hour:")
print(hour_analysis)

# 6. COMBINED PATTERN ANALYSIS
print(f"\n{'='*80}")
print("6. HIGH-RISK TRADE PATTERNS (Multiple Red Flags)")
print(f"{'='*80}\n")

# Define risk factors
df_all['high_rsi'] = df_all['entry_rsi'] > 28
df_all['low_atr'] = df_all['atr_pct'] < 2.0
df_all['late_entry'] = df_all['entry_hour'] >= 14  # After 2 PM

# Count risk factors
df_all['risk_factors'] = (
    df_all['high_rsi'].astype(int) + 
    df_all['low_atr'].astype(int) + 
    df_all['late_entry'].astype(int)
)

risk_analysis = df_all.groupby('risk_factors').agg({
    'pnl': ['count', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0, 'mean', 'sum']
}).round(2)
risk_analysis.columns = ['Trades', 'Win_Rate_%', 'Avg_PnL', 'Total_PnL']
print("Performance by Number of Risk Factors:")
print(risk_analysis)

# 7. ACTIONABLE RECOMMENDATIONS
print(f"\n{'='*80}")
print("7. ACTIONABLE RECOMMENDATIONS TO REDUCE LOSSES")
print(f"{'='*80}\n")

recommendations = []

# Ticker exclusion
if len(bad_tickers) > 0:
    potential_savings = worst_tickers[worst_tickers['Win_Rate'] < 40]['Net_PnL'].sum()
    recommendations.append({
        'action': f"Exclude {len(bad_tickers)} tickers with <40% win rate",
        'impact': f"Avoid ${abs(potential_savings):.2f} in losses",
        'tickers': bad_tickers[:10]
    })

# RSI threshold
high_rsi_impact = df_all[df_all['entry_rsi'] > 28]['pnl'].sum()
high_rsi_count = len(df_all[df_all['entry_rsi'] > 28])
if high_rsi_impact < 0:
    recommendations.append({
        'action': "Lower RSI entry threshold from 30 to 28",
        'impact': f"Avoid {high_rsi_count} trades, save ${abs(high_rsi_impact):.2f}",
        'details': f"Trades with RSI > 28 have lower win rate"
    })

# ATR threshold
low_atr_trades = df_all[df_all['atr_pct'] < 2.0]
if len(low_atr_trades) > 0 and low_atr_trades['pnl'].sum() < 0:
    recommendations.append({
        'action': "Increase min ATR% from 1.5% to 2.0%",
        'impact': f"Avoid {len(low_atr_trades)} low-volatility trades",
        'details': f"Low ATR trades underperform"
    })

# Time of day
late_trades = df_all[df_all['entry_hour'] >= 14]
if len(late_trades) > 0:
    late_wr = (late_trades['pnl'] > 0).sum() / len(late_trades) * 100
    if late_wr < 55:
        recommendations.append({
            'action': "Avoid entries after 2:00 PM",
            'impact': f"{len(late_trades)} trades with {late_wr:.1f}% win rate",
            'details': "Late entries have less time to recover"
        })

# High risk combinations
high_risk_trades = df_all[df_all['risk_factors'] >= 2]
if len(high_risk_trades) > 0:
    hr_pnl = high_risk_trades['pnl'].sum()
    if hr_pnl < 0:
        recommendations.append({
            'action': "Skip trades with 2+ risk factors",
            'impact': f"Avoid {len(high_risk_trades)} high-risk trades, save ${abs(hr_pnl):.2f}",
            'details': "Trades with multiple red flags rarely work"
        })

print(f"\nFound {len(recommendations)} actionable improvements:\n")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['action']}")
    print(f"   Impact: {rec['impact']}")
    if 'tickers' in rec:
        print(f"   Tickers: {', '.join(rec['tickers'])}")
    if 'details' in rec:
        print(f"   Details: {rec['details']}")
    print()

# 8. SAVE RESULTS
print(f"\n{'='*80}")
print("8. SAVING ANALYSIS RESULTS")
print(f"{'='*80}\n")

# Save bad tickers list
if len(bad_tickers) > 0:
    with open('bad_tickers_to_exclude.txt', 'w') as f:
        for ticker in bad_tickers:
            f.write(f"{ticker}\n")
    print(f"✅ Saved {len(bad_tickers)} bad tickers to bad_tickers_to_exclude.txt")

# Save recommendations
with open('loss_reduction_recommendations.txt', 'w') as f:
    f.write("LOSS REDUCTION RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    for i, rec in enumerate(recommendations, 1):
        f.write(f"{i}. {rec['action']}\n")
        f.write(f"   Impact: {rec['impact']}\n")
        if 'details' in rec:
            f.write(f"   Details: {rec['details']}\n")
        f.write("\n")
print("✅ Saved recommendations to loss_reduction_recommendations.txt")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
