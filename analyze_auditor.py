
import os
import pandas as pd

def main():
    fpath = "auditor_results_temp/auditor-state-20761271726/auditor_log.csv"
    print(f"Analyzing {fpath}")
    
    try:
        df = pd.read_csv(fpath)
        print("Columns found:", df.columns.tolist())
        print("First row:", df.iloc[0].to_dict())
        
        # Check if 'pnl' exists or if it's named differently (e.g. 'realized_pnl')
        # auditor_bot.py uses "pnl" key for exit.

            
        if 'pnl' in df.columns: 
             exits = df[df['event'] == 'SHADOW_EXIT']
             print(f"Shadow Exits: {len(exits)}")
             if not exits.empty:
                 total_pnl = exits['pnl'].sum()
                 print(f"Total Shadow PnL: ${total_pnl:.2f}")
                 print(exits[['symbol', 'bot_tag', 'pnl', 'reason']].to_string())
        else:
             exits = df[df['event'] == 'SHADOW_EXIT']
             print(f"Shadow Exits: {len(exits)} (No PnL column found - implies no exits ever logged)")
        # Filter SHADOW_ENTRY (Skipped by Bot, Taken by Auditor)
        entries = df[df['event'] == 'SHADOW_ENTRY']
        # Filter for today if needed (checking timestamp string)
        if 'ts' in entries.columns:
             entries_today = entries[entries['ts'].astype(str).str.contains('2026-01-06')]
             print(f"Shadow Entries Today (Skipped by Bot): {len(entries_today)}")
             if not entries_today.empty:
                 print(entries_today[['symbol', 'bot_tag', 'reason', 'ts']].head(10).to_string())
        else:
             print("Entries found (no ts filter):", len(entries))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
