"""SQuAD 2.0 scoring: normalization, EM, F1, and no-answer handling.

Convention: a prediction of "" means the model abstained ("no answer"),
and an unanswerable question has gold answers [].
"""
import re
import string
from collections import Counter

PUNCTUATION = set(string.punctuation)


def normalize_answer(s):
    """Lowercase, strip punctuation, remove articles (a/an/the), collapse whitespace.
    Same steps and order as the official SQuAD 2.0 evaluation script."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in PUNCTUATION)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction, gold_answers):
    """Return 1 if the prediction matches any gold answer after normalization, else 0.
    gold_answers is a list of strings; [] means the question is unanswerable."""
    # Drop junk references like "." that normalize to "" (as the official script does);
    # if nothing is left, the question is unanswerable and the only correct answer is "".
    gold_answers = [g for g in gold_answers if normalize_answer(g)] or [""]
    pred = normalize_answer(prediction)
    return max(int(pred == normalize_answer(ref)) for ref in gold_answers)


def _f1_single(prediction, reference):
    """Token-overlap F1 between one prediction and one reference."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        # Abstention on either side: full credit only if both are empty.
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction, gold_answers):
    """Best token-overlap F1 over the gold answers; [] means unanswerable."""
    gold_answers = [g for g in gold_answers if normalize_answer(g)] or [""]  # see exact_match
    return max(_f1_single(prediction, ref) for ref in gold_answers)


def _pct(values):
    """Mean of a list of 0/1 (or 0..1) values as a percentage; None if the list is empty."""
    return 100 * sum(values) / len(values) if values else None


def compute_metrics(predictions, gold_answers_list):
    """Score a full set of predictions.
    predictions: list of strings ("" = abstained).
    gold_answers_list: list of gold-answer lists ([] = unanswerable)."""
    has_ans, no_ans = [], []  # one (em, f1, abstained) tuple per question
    for pred, golds in zip(predictions, gold_answers_list, strict=True):
        abstained = normalize_answer(pred) == ""
        row = (exact_match(pred, golds), f1_score(pred, golds), abstained)
        (has_ans if golds else no_ans).append(row)

    everything = has_ans + no_ans
    return {
        "n_total": len(everything),
        "n_has_ans": len(has_ans),
        "n_no_ans": len(no_ans),
        "exact": _pct([em for em, _, _ in everything]),
        "f1": _pct([f1 for _, f1, _ in everything]),
        "HasAns_exact": _pct([em for em, _, _ in has_ans]),
        "HasAns_f1": _pct([f1 for _, f1, _ in has_ans]),
        "NoAns_accuracy": _pct([em for em, _, _ in no_ans]),
        "hallucination_rate": _pct([not ab for _, _, ab in no_ans]),
        "over_abstention_rate": _pct([ab for _, _, ab in has_ans]),
    }

if __name__ == "__main__":
    examples = [
        "Rollo.",
        "The Normans",
        "the first half of the 10th century",
        "  New   York ",
        "anthem",
        "U.S.",
        "Roman-Gaulish",
        "1990–2000",  # en dash, not a hyphen
        "",
    ]
    for s in examples:
        print(f"{s!r:40} -> {normalize_answer(s)!r}")

    em_cases = [
        ("Rollo.", ["Rollo", "Rollo"], 1),
        ("Rollo the Viking", ["Rollo"], 0),
        ("10th century", ["10th", "10th century", "the first half of the 10th century"], 1),
        ("", [], 1),          # correct abstention
        ("Rollo", [], 0),     # hallucination
        ("", ["Rollo"], 0),   # over-abstention
        ("", ["interventionism", "."], 0),  # junk "." reference must not reward abstaining
    ]
    print()
    for pred, golds, expected in em_cases:
        got = exact_match(pred, golds)
        assert got == expected, (pred, golds, got, expected)
        print(f"EM={got}  pred={pred!r:20} gold={golds}")
    print("all EM cases pass")

    f1_cases = [
        ("Rollo the Viking", ["Rollo"], 0.667),
        ("10th century", ["the first half of the 10th century"], 0.571),
        ("10th century", ["10th", "10th century", "the first half of the 10th century"], 1.0),
        ("Charles III", ["Rollo"], 0.0),
        ("", [], 1.0),          # correct abstention
        ("Rollo", [], 0.0),     # hallucination
        ("", ["Rollo"], 0.0),   # over-abstention
        ("", ["interventionism", "."], 0.0),  # junk "." reference must not reward abstaining
    ]
    print()
    for pred, golds, expected in f1_cases:
        got = round(f1_score(pred, golds), 3)
        assert got == expected, (pred, golds, got, expected)
        print(f"F1={got:.3f}  pred={pred!r:20} gold={golds}")
    print("all F1 cases pass")

    # Toy set: 3 answerable (right, partly right, over-abstained) + 3 unanswerable
    # (abstained, hallucinated, abstained). Expected values worked out by hand.
    toy_preds = ["Rollo", "Rollo the Viking", "", "", "Rollo", ""]
    toy_golds = [["Rollo"], ["Rollo"], ["France"], [], [], []]
    expected = {
        "n_total": 6, "n_has_ans": 3, "n_no_ans": 3,
        "exact": 50.0,                  # 3 of 6
        "f1": 61.11,                    # (1 + 0.667 + 0 + 1 + 0 + 1) / 6
        "HasAns_exact": 33.33,          # 1 of 3
        "HasAns_f1": 55.56,             # (1 + 0.667 + 0) / 3
        "NoAns_accuracy": 66.67,        # 2 of 3
        "hallucination_rate": 33.33,    # 1 of 3 unanswerable got an answer
        "over_abstention_rate": 33.33,  # 1 of 3 answerable got ""
    }
    results = compute_metrics(toy_preds, toy_golds)
    print()
    for key, value in results.items():
        assert round(value, 2) == expected[key], (key, value, expected[key])
        print(f"{key:22} {value:.2f}")
    print("all aggregate metrics pass")
