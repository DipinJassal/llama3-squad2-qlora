"""Prompt variants, output parsing, and the LLM-as-a-judge prompt.

Every prompt is a list of chat messages; vLLM applies LLaMA-3's chat template.
"""
import random
import re

ABSTAIN_TOKEN = "unanswerable"

ZERO_SHOT_SYSTEM = (
    "Answer the question using the passage. Respond with only the answer: "
    "a short span copied from the passage, no explanation."
)
ABSTAIN_SYSTEM = (
    ZERO_SHOT_SYSTEM + " Some questions cannot be answered from the passage. "
    f"If the passage does not contain the answer, respond with exactly: {ABSTAIN_TOKEN}"
)
# Self-check is a separate verification call, not a "check your answer" follow-up turn:
# the follow-up version made LLaMA apologize and retract correct answers (sycophancy) and
# ramble instead of repeating the span.
SELF_CHECK_SYSTEM = (
    "You verify answers for a reading-comprehension task. Given a passage, a question and a "
    "proposed answer, decide whether the passage actually supports the proposed answer as the "
    "answer to this exact question. Unanswerable questions often change one detail of the "
    "passage (a date, a name, a negation), so check that every detail of the question matches "
    "the passage. Reply with exactly one word: SUPPORTED or UNSUPPORTED."
)


def self_check_messages(context, question, answer):
    return [
        {"role": "system", "content": SELF_CHECK_SYSTEM},
        {"role": "user", "content": f"{user_message(context, question)}\n\nProposed answer: {answer}"},
    ]


def user_message(context, question):
    return f"Passage: {context}\n\nQuestion: {question}"


def pick_few_shot_examples(train, n_each=2, max_context_chars=700, seed=0):
    """Fixed few-shot demos from train: alternating answerable / unanswerable, short passages."""
    rng = random.Random(seed)
    short = [ex for ex in train.select(range(20000)) if len(ex["context"]) <= max_context_chars]
    answerable = [ex for ex in short if ex["answers"]["text"]]
    unanswerable = [ex for ex in short if not ex["answers"]["text"]]
    demos = []
    for a, u in zip(rng.sample(answerable, n_each), rng.sample(unanswerable, n_each)):
        demos += [(a, a["answers"]["text"][0]), (u, ABSTAIN_TOKEN)]
    return demos


def build_messages(variant, context, question, few_shot_demos=None):
    """First-turn messages for a prompt variant. (self_check's second turn is added later.)"""
    if variant == "zero_shot":
        system = ZERO_SHOT_SYSTEM
    elif variant in ("abstain", "few_shot", "self_check"):
        system = ABSTAIN_SYSTEM
    else:
        raise ValueError(f"unknown variant {variant!r}")
    messages = [{"role": "system", "content": system}]
    if variant == "few_shot":
        for ex, answer in few_shot_demos:
            messages += [
                {"role": "user", "content": user_message(ex["context"], ex["question"])},
                {"role": "assistant", "content": answer},
            ]
    messages.append({"role": "user", "content": user_message(context, question)})
    return messages


# Phrases that mean "no answer" when the model doesn't use the exact abstain token.
ABSTAIN_RE = re.compile(
    r"\bunanswerable\b|cannot be answered|can't be answered|not (be )?answerable"
    r"|\bno answer\b|\bno mention\b|not (mentioned|stated|specified|provided|given|found|included)"
    r"|does not (say|mention|state|specify|provide|contain|include|indicate)"
    r"|doesn't (say|mention|state|specify|provide|contain|include|indicate)"
    r"|\bno\b[^.]{0,80}\bmentioned\b"
    r"|not enough information|no information|\bn/a\b|^(unknown|none)\.?$",
    re.IGNORECASE,
)


def parse_answer(raw):
    """Rule-based parser: model text -> prediction string ("" = abstained)."""
    text = raw.strip()
    if ABSTAIN_RE.search(text):
        return ""
    text = text.splitlines()[0] if text else ""
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip()


# The judge sees only the response, not the question: with the question it tended to
# judge correctness (labelling wrong-looking answers ABSTAIN). See LEARNING_LOG Stage 5.
JUDGE_SYSTEM = (
    "You classify a single response from a reading-comprehension system as ABSTAIN or ANSWER.\n"
    "ABSTAIN: the response declines to answer, or says the information is not in the passage, "
    "is not mentioned, is unknown, or that the question cannot be answered. 'None' is ABSTAIN.\n"
    "ANSWER: anything else - any fact, name, number, phrase or sentence offered as an answer.\n\n"
    "Examples:\nRollo -> ANSWER\nThe passage does not say when the war ended. -> ABSTAIN\n"
    "They lost money from the beginning. -> ANSWER\nThere is no mention of gold prices. -> ABSTAIN\n"
    "now known as the Amazon Basin. -> ANSWER\nunanswerable -> ABSTAIN\n"
    "No, it was the network that handled delivery. -> ANSWER\n\n"
    "Reply with exactly one word: ABSTAIN or ANSWER."
)


def judge_messages(response):
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Response: {response}"},
    ]


def parse_judge(raw):
    """Judge text -> "ABSTAIN", "ANSWER", or "INVALID"."""
    text = raw.strip().upper()
    for label in ("ABSTAIN", "ANSWER"):
        if text.startswith(label):
            return label
    return "INVALID"
