import os
import json
import shutil
from RubberBand.src.ticker_health import TickerHealthManager

def test_resilience():
    # Setup
    test_file = "test_health.json"
    if os.path.exists(test_file):
        os.remove(test_file)
        
    config = {
        "enabled": True,
        "lookback_trades": 5,
        "max_consecutive_losses": 3,
        "drawdown_threshold_usd": -100.0,
        "probation_period_days": 7
    }
    
    mgr = TickerHealthManager(test_file, config)
    sym = "TEST_TICKER"
    
    print(f"Testing {sym}...")
    
    # 1. Initial State
    paused, reason = mgr.is_paused(sym)
    assert not paused, "Should be active initially"
    print("✅ Initial state: Active")
    
    # 2. Loss 1
    mgr.update_trade(sym, -10.0, "trade1")
    paused, _ = mgr.is_paused(sym)
    assert not paused, "Should be active after 1 loss"
    print("✅ After 1 loss: Active")
    
    # 3. Loss 2
    mgr.update_trade(sym, -10.0, "trade2")
    paused, _ = mgr.is_paused(sym)
    assert not paused, "Should be active after 2 losses"
    print("✅ After 2 losses: Active")
    
    # 4. Loss 3 (Trigger Streak)
    mgr.update_trade(sym, -10.0, "trade3")
    paused, reason = mgr.is_paused(sym)
    assert paused, "Should be PAUSED after 3 losses"
    print(f"✅ After 3 losses: PAUSED ({reason})")
    
    # 5. Reset
    mgr.reset_status(sym)
    paused, _ = mgr.is_paused(sym)
    assert not paused, "Should be active after reset"
    print("✅ After reset: Active")
    
    # 6. Drawdown Trigger
    mgr.update_trade(sym, -150.0, "trade4")
    paused, reason = mgr.is_paused(sym)
    assert paused, "Should be PAUSED after large drawdown"
    print(f"✅ After large loss (-$150): PAUSED ({reason})")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    print("\n🎉 All resilience tests passed!")

if __name__ == "__main__":
    test_resilience()
