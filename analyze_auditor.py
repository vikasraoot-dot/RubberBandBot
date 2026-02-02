import json
import glob

# Aggregate EOD summaries for both bots
def analyze_bot(pattern, name):
    files = glob.glob(pattern)
    daily_stats = {}
    all_trades = []
    
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                try:
                    d = json.loads(line.strip())
                    if d.get('type') == 'EOD_SUMMARY':
                        date = d.get('date', 'unknown')
                        # Only keep latest EOD for each date
                        daily_stats[date] = d
                    elif d.get('type') == 'ENTRY_ACK':
                        all_trades.append(d)
                except:
                    pass
    
    print(f'\n=== {name} ===')
    print(f'Trading Days: {len(daily_stats)}')
    print(f'Total Entries: {len(all_trades)}')
    
    if daily_stats:
        total_pnl = sum(d.get('total_pnl', 0) for d in daily_stats.values())
        total_closed = sum(d.get('closed_trades', 0) for d in daily_stats.values())
        total_wins = sum(d.get('win_count', 0) for d in daily_stats.values())
        total_losses = sum(d.get('loss_count', 0) for d in daily_stats.values())
        
        print(f'Closed Trades: {total_closed}')
        print(f'Wins: {total_wins}, Losses: {total_losses}')
        if total_closed > 0:
            print(f'Win Rate: {total_wins/total_closed*100:.1f}%')
        print(f'Total PnL: ${total_pnl:.2f}')
        
        # Show daily breakdown
        print('\nDaily PnL:')
        for date in sorted(daily_stats.keys())[-10:]:
            d = daily_stats[date]
            pnl = d.get('total_pnl', 0)
            entries = d.get('total_trades', 0)
            closed = d.get('closed_trades', 0)
            print(f"  {date}: PnL=${pnl:.2f}, Entries={entries}, Closed={closed}")

analyze_bot('auditor_logs/15M_STK_*.jsonl', '15M STOCK BOT')
analyze_bot('auditor_logs/15M_OPT_*.jsonl', '15M OPTIONS BOT')
