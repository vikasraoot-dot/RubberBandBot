import yaml
import os

path = "RubberBand/config.yaml"
print(f"Loading {path} from {os.getcwd()}")
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
    
print(f"Keys: {list(cfg.keys())}")
print(f"Trend Filter: {cfg.get('trend_filter')}")
print(f"Allow Shorts: {cfg.get('allow_shorts')}")
