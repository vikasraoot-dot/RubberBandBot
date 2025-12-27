
import os
import sys

# Add repo root
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.options_data import get_option_snapshot

def test_fixed_function():
    symbol = "CVNA251226C00455000"
    print(f"Testing get_option_snapshot for: {symbol}")
    
    snapshot = get_option_snapshot(symbol)
    if snapshot:
        print("SUCCESS: Snapshot retrieved!")
        print(f"Bid: {snapshot['bid']}, Ask: {snapshot['ask']}, Delta: {snapshot['delta']}, Theta: {snapshot['theta']}")
        # Verify greeks are floats
        if isinstance(snapshot['delta'], float) and isinstance(snapshot['theta'], float):
             print("Data type check passed: Greeks are floats.")
        else:
             print("Data type check failed!")
    else:
        print("FAILURE: returned None")

if __name__ == "__main__":
    print("--- Verifying Fix ---")
    test_fixed_function()
