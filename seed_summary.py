"""Run-to-run variance: mean / std / range over seeds 0-2 for the replicated configs."""
import json
import statistics
from pathlib import Path

CONFIGS = ["qv-r8", "all-r16", "all-r64"]
METRICS = ["exact", "f1", "HasAns_exact", "NoAns_accuracy", "hallucination_rate", "over_abstention_rate"]


def load(name):
    return json.loads(Path(f"results/{name}__abstain__full.json").read_text())["parse"]


for config in CONFIGS:
    runs = [load(config)] + [load(f"{config}-s{s}") for s in (1, 2)]   # seed 0 has no suffix
    print(f"\n{config}  (seeds 0, 1, 2; full dev, parser scoring)")
    print(f"  {'metric':22} {'seed0':>7} {'seed1':>7} {'seed2':>7} {'mean':>7} {'std':>6} {'range':>6}")
    for m in METRICS:
        vals = [r[m] for r in runs]
        print(f"  {m:22} " + " ".join(f"{v:7.2f}" for v in vals)
              + f" {statistics.mean(vals):7.2f} {statistics.stdev(vals):6.2f} {max(vals) - min(vals):6.2f}")
