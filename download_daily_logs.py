import os
import subprocess
import json
import datetime
import shutil
import zipfile
import argparse

# Configuration
# Default to today if not provided via args
DEFAULT_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

WORKFLOWS = {
    "15m_Stock": "rubberband-live-loop-am.yml",
    "15m_Options": "rubberband-options-spreads.yml",
    "Weekly_Stock": "weekly-stock-live.yml",
    "Weekly_Options": "weekly-options-live.yml"
}

def run_gh_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}\n{result.stderr}")
        return None
    return result.stdout.strip()

def main():
    parser = argparse.ArgumentParser(description="Download daily logs from GitHub Actions")
    parser.add_argument("--date", type=str, default=DEFAULT_DATE, help="Date to search for (YYYY-MM-DD)")
    args = parser.parse_args()
    
    target_date = args.date
    # Normalize date format for filename (remove dashes for YYYYMMDD)
    date_compact = target_date.replace("-", "")
    
    base_dir = f"latest runs/{target_date}"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")

    print(f"--- Searching for runs on {target_date} ---")

    for bot_name, workflow in WORKFLOWS.items():
        print(f"\nChecking {bot_name} ({workflow})...")
        
        # List runs (JSON)
        cmd = f'gh run list --workflow {workflow} --limit 5 --json databaseId,createdAt,status,conclusion'
        output = run_gh_cmd(cmd)
        
        if not output:
            continue
            
        runs = json.loads(output)
        target_run = None
        
        # Find today's run
        for run in runs:
            # createdAt format: "2025-12-19T14:30:00Z"
            created_at = run['createdAt']
            if target_date in created_at:
                target_run = run
                break
        
        if target_run:
            run_id = target_run['databaseId']
            status = target_run['status']
            conclusion = target_run['conclusion']
            print(f"  Found Run ID: {run_id} | Status: {status} | Conclusion: {conclusion}")
            
            # Download
            bot_dir = os.path.join(base_dir, bot_name)
            if not os.path.exists(bot_dir):
                os.makedirs(bot_dir)
                
            print(f"  Downloading artifacts to {bot_dir}...")
            # gh run download <ID> -D <DIR>
            dl_cmd = f'gh run download {run_id} -D "{bot_dir}"'
            run_gh_cmd(dl_cmd)
            
            # Unzip recursively and RENAME console.log
            print("  Unzipping and processing logs...")
            for root, dirs, files in os.walk(bot_dir):
                for file in files:
                    if file.endswith(".zip"):
                        zip_path = os.path.join(root, file)
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(root)
                            print(f"    Extracted: {file}")
                            # Clean up zip
                            os.remove(zip_path) 
                        except Exception as e:
                            print(f"    Error extracting {file}: {e}")
            
            # Check for console.log and rename it
            console_log_path = os.path.join(bot_dir, "console.log")
            if os.path.exists(console_log_path):
                new_name = f"logs_{bot_name.lower()}_{date_compact}.txt"
                new_path = os.path.join(bot_dir, new_name)
                # Copy/Rename logic
                # We often want the file in the ROOT of the daily folder for easier access, 
                # or keep it in the bot folder. Let's keep it in the bot folder but renamed.
                shutil.move(console_log_path, new_path)
                print(f"    RENAMED: console.log -> {new_name}")
                
                # OPTIONAL: Copy to root for convenience (matching user's previous structure?)
                # root_copy = os.path.join("logs_downloaded", new_name)
                # os.makedirs("logs_downloaded", exist_ok=True)
                # shutil.copy(new_path, root_copy)
                
            else:
                print(f"    [WARN] No console.log found in {bot_dir}")
                            
        else:
            print(f"  [WARN] No run found for date {target_date}")

    print("\nDone.")

if __name__ == "__main__":
    main()
