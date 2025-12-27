
import random

# Load candidates
with open("RubberBand/candidates_tickers.txt", "r") as f:
    lines = [l.strip() for l in f if l.strip()]

# Sample 50 random tickers (plus ensure some known high vol ones are there)
known_vol = ["NVDA", "MSTR", "SOXL", "TQQQ", "SMH", "AMD"]
sample = random.sample(lines, 45) + known_vol
unique_sample = list(set(sample))

with open("RubberBand/tickers_exhaustive_sample.txt", "w") as f:
    for s in unique_sample:
        f.write(f"{s}\n")

print(f"Created sample of {len(unique_sample)} tickers.")
