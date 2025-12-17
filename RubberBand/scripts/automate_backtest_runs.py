import os
import sys
import json
import time
import subprocess
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to GitHub CLI - verified from previous steps
GH_EXE = r"C:\Program Files\GitHub CLI\gh.exe"
REPO = "vikasraoot-dot/RubberBandBot"  # Update if needed, or rely on local git context

# Base Output Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LATEST_RUNS_DIR = os.path.join(BASE_DIR, "latest runs")

# ==============================================================================
# HELPERS
# ==============================================================================

def run_command(args: List[str], check=True) -> subprocess.CompletedProcess:
    """Run a subprocess command using the specific GH executable."""
    # If the command starts with 'gh', replace it with full path
    if args[0] == "gh":
        args[0] = GH_EXE
    
    # print(f"DEBUG: Running: {' '.join(args)}")
    try:
        # Use shell=True only if needed, but avoiding it is safer. 
        # However, purely executing the exe should be fine.
        result = subprocess.run(
            args, 
            capture_output=True, 
            text=True, 
            check=check,
            cwd=BASE_DIR,  # Run from repo root
            encoding='utf-8' # Force Verify UTF-8
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR executing command: {' '.join(args)}")
        print(f"STDERR: {e.stderr}")
        raise e

def get_latest_run_id(workflow_file: str) -> str:
    """Get the most recent run ID for a workflow to detect the new trigger."""
    # gh run list --workflow <file> --limit 1 --json databaseId
    res = run_command(["gh", "run", "list", "--workflow", workflow_file, "--limit", "1", "--json", "databaseId"])
    data = json.loads(res.stdout)
    if not data:
        return None
    return str(data[0]["databaseId"])

def trigger_workflow(name: str, workflow_file: str, inputs: Dict[str, str]) -> str:
    """
    Trigger a workflow and return its (presumed) new run ID.
    Note: GH CLI doesn't return the Run ID immediately. We have to poll for it.
    Strategy: Get latest ID -> Trigger -> Wait for new ID > latest ID.
    """
    print(f"[{name}] Triggering {workflow_file}...")
    
    # Get current latest ID to distinguish the new one
    old_id = get_latest_run_id(workflow_file)
    
    # Construct args
    cmd = ["gh", "workflow", "run", workflow_file]
    for k, v in inputs.items():
        cmd.extend(["-f", f"{k}={v}"])
        
    run_command(cmd)
    
    # Wait for new run to appear
    print(f"[{name}] Waiting for run to start...", end="", flush=True)
    for _ in range(20): # Try for 60 seconds (3s sleep)
        time.sleep(3)
        print(".", end="", flush=True)
        new_id = get_latest_run_id(workflow_file)
        if new_id != old_id:
            print(f" Started! (ID: {new_id})")
            return new_id
            
    raise TimeoutError(f"Workflow {workflow_file} triggered but did not appear in run list.")

def wait_for_completion(run_id: str, name: str) -> str:
    """Poll run status until complete."""
    print(f"[{name}] Monitor Run {run_id}...", end="", flush=True)
    while True:
        # gh run view <id> --json status,conclusion
        try:
            res = run_command(["gh", "run", "view", run_id, "--json", "status,conclusion"], check=False)
            if res.returncode != 0:
                print("?", end="", flush=True)
                time.sleep(5)
                continue
                
            data = json.loads(res.stdout)
            status = data.get("status")
            conclusion = data.get("conclusion")
            
            if status == "completed":
                print(f" Done! ({conclusion})")
                return conclusion
            
            print(".", end="", flush=True)
            time.sleep(10)
        except Exception as e:
            print(f"ERROR polling: {e}")
            time.sleep(10)

def download_results(run_id: str, output_dir: str):
    """Download artifacts to specific directory."""
    os.makedirs(output_dir, exist_ok=True)
    # gh run download <id> --dir <path>
    run_command(["gh", "run", "download", run_id, "--dir", output_dir])

# ==============================================================================
# REPORTING
# ==============================================================================

def generate_text_report(csv_path: str, title: str, config: Dict = None) -> str:
    """Generate the user-requested text block report from a trade CSV."""
    if not os.path.exists(csv_path):
        return f"Error: CSV not found at {csv_path}"

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return f"Error reading CSV: {csv_path}"
    
    total_trades = len(df)
    total_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0
    
    # Cost
    if 'cost' in df.columns:
        total_cost = df['cost'].sum()
    elif 'entry_debit' in df.columns:
        total_cost = (df['entry_debit'] * 100).sum()
    else:
        total_cost = 0 # Cannot calc ROI properly
        
    roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    win_count = len(wins)
    loss_count = check_len = len(losses)
    
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    
    avg_bars = df['bars_held'].mean() if 'bars_held' in df.columns else 0
    
    # Exits
    exits = df['reason'].value_counts() if 'reason' in df.columns else {}
    tp = exits.get('MAX_PROFIT', 0)
    sl = exits.get('STOP_LOSS', 0)
    time_exit = exits.get('BARS_STOP', 0) + exits.get('EOD', 0)
    
    # Config string
    spread_cfg = "Unknown"
    if config:
        spread_cfg = f"DTE={config.get('dte')}, Width={config.get('spread_width')} ATR, ADX={config.get('adx_max')}, SL={config.get('sl_pct')}"

    return f"""============================================================
{title}
============================================================
Period: {config.get('days') if config else '?'} days
Spread Config: {spread_cfg}
------------------------------------------------------------
Total Trades: {total_trades}
Total Cost: ${total_cost:,.2f}
Total P&L: ${total_pnl:,.2f}
ROI: {roi:.1f}%
Win Rate: {win_rate:.1f}% ({win_count}W / {loss_count}L)
Avg Win: ${avg_win:.2f}
Avg Loss: ${avg_loss:.2f}
Avg Bars Held: {avg_bars:.1f}
------------------------------------------------------------
Exit @ Max Profit: {tp} ({tp/total_trades*100 if total_trades else 0:.1f}%)
Exit @ Max Loss: {sl} ({sl/total_trades*100 if total_trades else 0:.1f}%)
Exit @ Time Limit: {time_exit} ({time_exit/total_trades*100 if total_trades else 0:.1f}%)
============================================================
"""

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Automate RubberBandBot Backtests")
    parser.add_argument("--config", help="Path to JSON config file with experiments")
    parser.add_argument("--auto-defaults", action="store_true", help="Run default set of experiments if no config provided")
    args = parser.parse_args()

    # 1. Load Experiments
    experiments = []
    
    if args.config:
        with open(args.config, "r") as f:
            experiments = json.load(f)
    elif args.auto_defaults:
        # Define default experiments: ADX A/B Test
        experiments = [
            {
                "name": "Live_baseline", 
                "workflow": "15m-options-backtest.yml",
                "inputs": {
                    "days": "30",
                    "adx_max": "0",
                    "slope_threshold": "-0.20",
                    "sl_pct": "0.80"
                }
            },
            {
                "name": "With_ADX_60", 
                "workflow": "15m-options-backtest.yml",
                "inputs": {
                    "days": "30",
                    "adx_max": "60",
                    "slope_threshold": "-0.20",
                    "sl_pct": "0.80"
                }
            }
        ]
    else:
        print("Please provide --config <file> or use --auto-defaults")
        return

    # 2. Setup Output Directory
    today_str = datetime.now().strftime("%m_%d_%Y")
    base_output = os.path.join(LATEST_RUNS_DIR, today_str, "BacktestRuns")
    os.makedirs(base_output, exist_ok=True)
    
    print(f"Target Output: {base_output}")
    print(f"Queuing {len(experiments)} experiments...")

    # 3. Trigger Loop
    results_to_csv = []
    full_text_report = ""
    
    for exp in experiments:
        name = exp["name"]
        wf = exp["workflow"]
        inputs = exp["inputs"]
        
        try:
            # Trigger
            run_id = trigger_workflow(name, wf, inputs)
            
            # Wait
            conclusion = wait_for_completion(run_id, name)
            
            # Download
            if conclusion == "success":
                exp_dir = os.path.join(base_output, f"{name}_{run_id}")
                print(f"[{name}] Downloading artifacts to {exp_dir}...")
                download_results(run_id, exp_dir)
                
                # Locate results CSV for analysis
                csv_file = None
                is_stock_bot = "stock" in inputs.get("bot_type", "").lower() or "backtest.yml" in wf # Detection heuristic
                
                target_csv = "spread_backtest_trades.csv"
                if "rubberband-backtest.yml" in wf:
                    target_csv = "detailed_trades.csv"
                    
                for root, dirs, files in os.walk(exp_dir):
                    if target_csv in files:
                        csv_file = os.path.join(root, target_csv)
                        break
                
                # Generate Block Report
                if csv_file:
                    block = generate_text_report(csv_file, name.upper(), inputs)
                    print("\n" + block)
                    full_text_report += block + "\n\n"
                else:
                    print(f"[{name}] Warning: {target_csv} not found in artifacts.")
                
                results_to_csv.append({
                    "experiment": name,
                    "run_id": run_id,
                    "conclusion": conclusion,
                    "download_path": exp_dir
                })
            else:
                print(f"[{name}] Run failed. Skipping download.")
                results_to_csv.append({
                    "experiment": name,
                    "run_id": run_id,
                    "conclusion": conclusion
                })
                
        except Exception as e:
            print(f"[{name}] Critical Error: {e}")
    
    # 4. Save Outputs
    if results_to_csv:
        # Save CSV summary
        df = pd.DataFrame(results_to_csv)
        csv_path = os.path.join(base_output, "automation_report.csv")
        df.to_csv(csv_path, index=False)
        
        # Save Text Report
        txt_path = os.path.join(base_output, "detailed_report.txt")
        with open(txt_path, "w") as f:
            f.write(full_text_report)
            
        print(f"\nSaved CSV Report to: {csv_path}")
        print(f"Saved Text Report to: {txt_path}")

if __name__ == "__main__":
    main()
