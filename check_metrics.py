"""Cross-check our metrics.py against Hugging Face's official squad_v2 metric."""
import random

import evaluate
from datasets import load_dataset

from metrics import compute_metrics

official = evaluate.load("squad_v2")

# Our key -> the official metric's key for the same quantity
KEY_MAP = {
    "exact": "exact",
    "f1": "f1",
    "HasAns_exact": "HasAns_exact",
    "HasAns_f1": "HasAns_f1",
    "NoAns_accuracy": "NoAns_exact",
}


def compare(name, ids, predictions, gold_answers_list):
    ours = compute_metrics(predictions, gold_answers_list)
    theirs = official.compute(
        predictions=[
            {"id": i, "prediction_text": p, "no_answer_probability": 0.0}
            for i, p in zip(ids, predictions)
        ],
        references=[
            {"id": i, "answers": {"text": g, "answer_start": [0] * len(g)}}
            for i, g in zip(ids, gold_answers_list)
        ],
    )
    print(f"\n{name} ({len(ids)} questions)")
    print(f"{'metric':16} {'ours':>9} {'official':>9}")
    for our_key, their_key in KEY_MAP.items():
        a, b = ours[our_key], theirs[their_key]
        status = "OK" if abs(a - b) < 1e-6 else "MISMATCH"
        print(f"{our_key:16} {a:9.4f} {b:9.4f}  {status}")


# 1. The hand-checked toy set from metrics.py
compare(
    "toy set",
    [str(i) for i in range(6)],
    ["Rollo", "Rollo the Viking", "", "", "Rollo", ""],
    [["Rollo"], ["Rollo"], ["France"], [], [], []],
)

# 2. Full dev set with synthetic predictions covering every case
dev = load_dataset("rajpurkar/squad_v2")["validation"]
rng = random.Random(0)
all_golds = [ex["answers"]["text"] for ex in dev]
answer_pool = [g[0] for g in all_golds if g]

predictions = []
for ex, golds in zip(dev, all_golds):
    if golds:
        gold = rng.choice(golds)
        kind = rng.choice(["exact", "case_punct", "partial", "wrong", "abstain"])
        if kind == "exact":
            predictions.append(gold)
        elif kind == "case_punct":
            predictions.append(gold.upper() + ".")
        elif kind == "partial":
            predictions.append(gold + " " + rng.choice(ex["context"].split()))
        elif kind == "wrong":
            predictions.append(rng.choice(answer_pool))
        else:
            predictions.append("")
    else:
        if rng.random() < 0.5:
            predictions.append("")
        else:
            words = ex["context"].split()
            start = rng.randrange(len(words))
            predictions.append(" ".join(words[start:start + rng.randint(1, 4)]))

compare("dev set, synthetic predictions", dev["id"], predictions, all_golds)
