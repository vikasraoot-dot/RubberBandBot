import json
import subprocess
import os
import shutil

def get_runs(workflow_name, limit=5):
    cmd = ["gh", "run", "list", "--workflow", workflow_name, "--limit", str(limit), "--json", "databaseId,status,conclusion,createdAt", "--status", "success"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        # Sort by createdAt descending
        data.sort(key=lambda x: x['createdAt'], reverse=True)
        return data
    except Exception as e:
        print(f"Error parsing runs for {workflow_name}: {e}")
        return []

def download_summary(run_id, output_dir):
    # Download specific artifact 'backtest-results'
    # gh run download <id> -n backtest-results -D <dir>
    cmd = ["gh", "run", "download", str(run_id), "-n", "backtest-results", "-D", output_dir]
    subprocess.run(cmd, check=True, capture_output=True)

def analyze_backtest_results():
    os.makedirs("analysis_temp", exist_ok=True)
    
    # 1. Stock Runs
    print("Fetching Stock Runs...")
    stock_runs = get_runs("rubberband-backtest.yml", limit=10)
    # Check for recent runs (last 1 hour)
    recent_stock = [r for r in stock_runs] # Filter by time if needed, for now take top 4
    
    results = []
    
    for i, run in enumerate(recent_stock[:4]):
        run_id = run['databaseId']
        dir_name = f"analysis_temp/stock_{run_id}"
        if os.path.exists(dir_name): shutil.rmtree(dir_name)
        
        print(f"Downloading Stock Run {run_id}...")
        try:
            download_summary(run_id, dir_name)
            
            # Read Summary
            summary_path = os.path.join(dir_name, "backtest_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    
                    # Handle V3: List of results
                    if isinstance(data, list):
                        total_pnl = sum(d.get("net", 0) for d in data)
                        total_trades = sum(d.get("trades", 0) for d in data)
                        # Calc average win rate weighted by trades? Or just sum wins?
                        # We don't have raw wins in this simple JSON, only win_rate.
                        # Approximate:
                        # wins = trades * win_rate
                        total_wins = sum(d.get("trades", 0) * d.get("win_rate", 0) for d in data)
                        win_rate = (total_wins / total_trades) if total_trades > 0 else 0
                        
                        # Days? Take max of days, or distinct.
                        # If data has 'days' key for each ticker.
                        days = max([d.get("days", 0) for d in data] or [0])
                        
                        results.append({
                            "type": "Stock",
                            "id": run_id,
                            "days": days,
                            "pnl": total_pnl,
                            "win_rate": win_rate,
                            "trades": total_trades,
                            "config": {} # config not saved in list format
                        })
                    
                    # Handle V2: Dict with metrics
                    elif isinstance(data, dict):
                         days = data.get("days", "Unknown")
                         total_pnl = data.get("total_pnl", 0)
                         win_rate = data.get("win_rate", 0)
                         
                         if "metrics" in data:
                              total_pnl = data["metrics"].get("total_pnl", 0)
                              win_rate = data["metrics"].get("win_rate", 0)

                         results.append({
                            "type": "Stock",
                            "id": run_id,
                            "days": days,
                            "pnl": total_pnl,
                            "win_rate": win_rate,
                            "trades": "N/A",
                            "config": data.get("config", {})
                         })
        except Exception as e:
            print(f"Failed to download/parse {run_id}: {e}")

    # 2. Options Runs
    print("Fetching Options Runs...")
    opt_runs = get_runs("15m-options-backtest.yml", limit=5)
    for i, run in enumerate(opt_runs[:4]):
        run_id = run['databaseId']
        dir_name = f"analysis_temp/opt_{run_id}"
        if os.path.exists(dir_name): shutil.rmtree(dir_name)
        
        print(f"Downloading Options Run {run_id}...")
        try:
            download_summary(run_id, dir_name)
             # Read Summary
            summary_path = os.path.join(dir_name, "backtest_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    
                    # Similar handling for Options if list
                    pnl = 0
                    wr = 0
                    tr = 0
                    d_val = "Unknown"
                    
                    if isinstance(data, list):
                        pnl = sum(d.get("net", 0) for d in data)
                        tr = sum(d.get("trades", 0) for d in data)
                        wins = sum(d.get("trades", 0) * d.get("win_rate", 0) for d in data) # Approx
                        wr = (wins / tr) if tr > 0 else 0
                        d_val = max([d.get("days", 0) for d in data] or [0])
                    else:
                        pnl = data.get("total_pnl", 0)
                        wr = data.get("win_rate", 0)
                        d_val = data.get("days", "Unknown")

                    results.append({
                        "type": "Options",
                        "id": run_id,
                        "days": d_val,
                        "pnl": pnl,
                        "win_rate": wr,
                        "trades": tr,
                        "config": data.get("config", {}) if isinstance(data, dict) else {}
                    })
        except Exception as e:
            print(f"Failed to download/parse {run_id}: {e}")

    # Print Report
    print("\n" + "="*60)
    print(f"{'Type':<10} {'Days':<10} {'PnL':<15} {'Win Rate':<10} {'Trades':<10}")
    print("-" * 60)
    
    results.sort(key=lambda x: (x['type'], str(x['days'])))
    
    for r in results:
        days = str(r['days'])
        pnl = f"${r['pnl']:.2f}"
        wr = f"{r['win_rate']:.2f}%" if isinstance(r['win_rate'], (int, float)) else str(r['win_rate'])
        tr = str(r.get('trades', 'N/A'))
        
        print(f"{r['type']:<10} {days:<10} {pnl:<15} {wr:<10} {tr:<10}")

if __name__ == "__main__":
    analyze_backtest_results()
