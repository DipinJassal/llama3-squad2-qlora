from datasets import load_dataset

ds = load_dataset("rajpurkar/squad_v2")
print(ds)


ex = ds["train"][0]
for key, value in ex.items():
    print(f"\n{key}: {value}")

def count_answerable(split):
    """Return (n_answerable, n_unanswerable) for a dataset split, e.g. ds["validation"]."""
    answers = split["answers"]
    n_answerable = sum(1 for a in answers if len(a["text"]) > 0)
    return n_answerable, len(answers) - n_answerable


for name in ["train", "validation"]:
    n_ans, n_unans = count_answerable(ds[name])
    total = n_ans + n_unans
    print(f"\n{name}: {total} total | {n_ans} answerable | {n_unans} unanswerable ({n_unans / total:.1%})")

def show_passage(split, idx):
    """Print one passage and every question asked about it, with its answers."""
    context = split[idx]["context"]
    print(f"\n=== {split[idx]['title']} ===\n{context}\n")
    for ex in split.filter(lambda ex: ex["context"] == context):
        answers = ex["answers"]["text"]
        label = sorted(set(answers)) if answers else "*** UNANSWERABLE ***"
        print(f"- {ex['question']}\n    -> {label}")


show_passage(ds["validation"], 0)
