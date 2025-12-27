import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

def generate_charts():
    # Find the CSV file
    csv_path = None
    for root, dirs, files in os.walk("artifacts/stock-180d-rsi25"):
        for f in files:
            if f == "detailed_trades.csv":
                csv_path = os.path.join(root, f)
                break
    
    if not csv_path:
        print("No data found!")
        return
    
    df = pd.read_csv(csv_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['date'] = df['exit_time'].dt.date
    df['is_winner'] = df['pnl'] > 0
    
    # Calculate stats
    total_trades = len(df)
    winners = len(df[df['pnl'] > 0])
    losers = total_trades - winners
    win_rate = winners / total_trades * 100
    total_pnl = df['pnl'].sum()
    
    print("\n" + "="*70)
    print("180-DAY RSI 25 BACKTEST RESULTS")
    print("="*70)
    print(f"Total Trades: {total_trades}")
    print(f"Winners: {winners} ({win_rate:.1f}%)")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Avg PnL: ${df['pnl'].mean():.2f}")
    print(f"Avg Win: ${df[df['pnl'] > 0]['pnl'].mean():.2f}")
    print(f"Avg Loss: ${df[df['pnl'] <= 0]['pnl'].mean():.2f}")
    print("="*70)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Stock Bot 180-Day Backtest: RSI 25 | {total_trades} Trades | Win Rate: {win_rate:.1f}% | PnL: ${total_pnl:,.2f}', 
                 fontsize=14, fontweight='bold')
    
    # --- Chart 1: Trades Over Time (Cumulative PnL) ---
    ax1 = axes[0]
    df_sorted = df.sort_values('exit_time')
    df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
    
    ax1.plot(df_sorted['exit_time'], df_sorted['cumulative_pnl'], 'b-', linewidth=1.5)
    ax1.fill_between(df_sorted['exit_time'], 0, df_sorted['cumulative_pnl'], 
                     where=df_sorted['cumulative_pnl'] >= 0, alpha=0.3, color='green')
    ax1.fill_between(df_sorted['exit_time'], 0, df_sorted['cumulative_pnl'], 
                     where=df_sorted['cumulative_pnl'] < 0, alpha=0.3, color='red')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_title('Cumulative P&L Over Time')
    ax1.set_ylabel('Cumulative PnL ($)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.grid(True, alpha=0.3)
    
    # --- Chart 2: Trade Count by Week (Stacked Win/Loss) ---
    ax2 = axes[1]
    df['week'] = df['exit_time'].dt.to_period('W').dt.start_time
    weekly = df.groupby(['week', 'is_winner']).size().unstack(fill_value=0)
    if True in weekly.columns and False in weekly.columns:
        weekly.rename(columns={True: 'Winners', False: 'Losers'}, inplace=True)
        weekly[['Winners', 'Losers']].plot(kind='bar', stacked=True, ax=ax2, 
                                             color=['green', 'red'], alpha=0.7, width=0.8)
    else:
        weekly.plot(kind='bar', ax=ax2, color='blue', alpha=0.7)
    
    ax2.set_title('Weekly Trade Distribution (Winners vs Losers)')
    ax2.set_ylabel('Number of Trades')
    ax2.set_xlabel('')
    ax2.legend(loc='upper right')
    # Format x-axis labels
    labels = [d.strftime('%b %d') for d in weekly.index]
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # --- Chart 3: Hold Duration Distribution ---
    ax3 = axes[2]
    
    # Convert hold_duration_bars to readable format
    hold_hours = df['hold_duration_bars'] * 0.25  # 15min bars = 0.25 hours
    
    # Create histogram
    bins = [0, 0.5, 1, 2, 4, 8, 24, 48, 100]
    labels_x = ['<30m', '30m-1h', '1-2h', '2-4h', '4-8h', '8-24h', '1-2d', '>2d']
    
    winners_hold = hold_hours[df['is_winner']]
    losers_hold = hold_hours[~df['is_winner']]
    
    hist_data = []
    winner_counts = []
    loser_counts = []
    for i in range(len(bins)-1):
        w_count = ((winners_hold >= bins[i]) & (winners_hold < bins[i+1])).sum()
        l_count = ((losers_hold >= bins[i]) & (losers_hold < bins[i+1])).sum()
        winner_counts.append(w_count)
        loser_counts.append(l_count)
    
    x = range(len(labels_x))
    width = 0.35
    ax3.bar([i - width/2 for i in x], winner_counts, width, label='Winners', color='green', alpha=0.7)
    ax3.bar([i + width/2 for i in x], loser_counts, width, label='Losers', color='red', alpha=0.7)
    
    ax3.set_title('Hold Duration Distribution (Winners vs Losers)')
    ax3.set_ylabel('Number of Trades')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_x)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add average hold duration text
    avg_hold_winners = winners_hold.mean()
    avg_hold_losers = losers_hold.mean()
    ax3.text(0.98, 0.95, f'Avg Hold (Win): {avg_hold_winners:.1f}h\nAvg Hold (Loss): {avg_hold_losers:.1f}h', 
             transform=ax3.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save chart
    chart_path = 'artifacts/stock_180d_rsi25_analysis.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Chart saved to: {chart_path}")
    plt.close()
    
    return chart_path

if __name__ == "__main__":
    generate_charts()
