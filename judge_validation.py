"""Validate the LLM judge against human labels (abstain vs answer).

    python judge_validation.py sample   # draw 250 responses -> labels/sample.csv (blind)
    python judge_validation.py label    # label them in the terminal -> labels/human_labels.csv
    python judge_validation.py score    # agreement + Cohen's kappa -> results/judge_validation.json

The sample excludes the 200 dev questions used while developing the judge and parser
(dev.shuffle(seed=0)[:200]); those would make the judge look better than it is.
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

SOURCES = {  # 50 responses from each: the 4 prompt variants + the best fine-tuned config
    "zero_shot": "outputs/llama3-8b-instruct__zero_shot__full.jsonl",
    "abstain": "outputs/llama3-8b-instruct__abstain__full.jsonl",
    "few_shot": "outputs/llama3-8b-instruct__few_shot__full.jsonl",
    "self_check": "outputs/llama3-8b-instruct__self_check__full.jsonl",
    "all-r16": "outputs/all-r16__abstain__full.jsonl",
}
PER_SOURCE = 50
SAMPLE, KEY, HUMAN = Path("labels/sample.csv"), Path("labels/key.csv"), Path("labels/human_labels.csv")


def dev_ids_used_in_development():
    from datasets import load_dataset
    dev = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(seed=0)
    return set(dev.select(range(200))["id"])


def sample():
    excluded = dev_ids_used_in_development()
    rng = random.Random(0)
    rows, key = [], []
    for source, path in SOURCES.items():
        recs = [json.loads(line) for line in open(path, encoding="utf-8")]
        recs = [r for r in recs if r["id"] not in excluded]
        for r in rng.sample(recs, PER_SOURCE):
            # self_check: the judge graded the final text, which is the verifier's "unanswerable"
            # when it rejected; label the text the judge actually saw.
            rows.append({"source": source, "id": r["id"], "question": r["question"], "response": r["raw"]})
            key.append({"judge": r["judge_label"], "parser": "ABSTAIN" if r["pred_parse"] == "" else "ANSWER",
                        "gold_unanswerable": int(not r["gold"])})
    order = list(range(len(rows)))
    rng.shuffle(order)  # mix sources so the labeller can't tell which prompt produced a response
    Path("labels").mkdir(exist_ok=True)
    for path, data, fields in [(SAMPLE, rows, ["idx", "source", "id", "question", "response"]),
                               (KEY, key, ["idx", "judge", "parser", "gold_unanswerable"])]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for new_idx, old in enumerate(order):
                w.writerow({"idx": new_idx, **data[old]})
    print(f"wrote {len(rows)} rows to {SAMPLE} (key kept separately in {KEY})")


def label():
    rows = list(csv.DictReader(open(SAMPLE, encoding="utf-8")))
    done = {}
    if HUMAN.exists():
        done = {r["idx"]: r["human"] for r in csv.DictReader(open(HUMAN, encoding="utf-8"))}
    print("For each response: does the system ANSWER or ABSTAIN (decline / say it's not in the passage)?")
    print("Don't judge correctness. Keys: a = ANSWER, b = ABSTAIN, u = unclear, q = save & quit\n")
    with open(HUMAN, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "human"])
        if not done:
            w.writeheader()
        for r in rows:
            if r["idx"] in done:
                continue
            print(f"[{int(r['idx']) + 1}/{len(rows)}]  Q: {r['question']}\n    Response: {r['response']}")
            while (k := input("    a/b/u/q > ").strip().lower()) not in {"a", "b", "u", "q"}:
                pass
            if k == "q":
                print("saved; run `label` again to continue")
                return
            w.writerow({"idx": r["idx"], "human": {"a": "ANSWER", "b": "ABSTAIN", "u": "UNCLEAR"}[k]})
            f.flush()
            print()
    print("all labelled - now run: python judge_validation.py score")


def cohen_kappa(a, b):
    n = len(a)
    p_obs = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_exp = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n**2
    return (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else float("nan"), p_obs


def score():
    rows = {r["idx"]: r for r in csv.DictReader(open(SAMPLE, encoding="utf-8"))}
    key = {r["idx"]: r for r in csv.DictReader(open(KEY, encoding="utf-8"))}
    human = {r["idx"]: r["human"] for r in csv.DictReader(open(HUMAN, encoding="utf-8"))}
    usable = [i for i, h in human.items() if h != "UNCLEAR"]
    h = [human[i] for i in usable]
    out = {"n_labelled": len(human), "n_unclear": len(human) - len(usable), "n_scored": len(usable)}
    for rater in ("judge", "parser"):
        r = [key[i][rater] for i in usable]
        kappa, agree = cohen_kappa(h, r)
        confusion = Counter(f"human={x} {rater}={y}" for x, y in zip(h, r))
        out[rater] = {"agreement": round(100 * agree, 2), "cohen_kappa": round(kappa, 4),
                      "confusion": dict(sorted(confusion.items()))}
        print(f"{rater:6} vs human: agreement {100 * agree:.2f}%   Cohen's kappa {kappa:.3f}")
        for k, v in sorted(confusion.items()):
            print(f"         {k:34} {v}")
    by_source = {}
    for i in usable:
        s = rows[i]["source"]
        by_source.setdefault(s, []).append(human[i] == key[i]["judge"])
    out["judge_agreement_by_source"] = {s: round(100 * sum(v) / len(v), 2) for s, v in by_source.items()}
    print("judge agreement by source:", out["judge_agreement_by_source"])
    disagreements = [{"question": rows[i]["question"], "response": rows[i]["response"],
                      "human": human[i], "judge": key[i]["judge"]}
                     for i in usable if human[i] != key[i]["judge"]]
    out["judge_disagreements"] = disagreements
    Path("results/judge_validation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{len(disagreements)} judge disagreements saved to results/judge_validation.json")


if __name__ == "__main__":
    {"sample": sample, "label": label, "score": score}[sys.argv[1]]()
