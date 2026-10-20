"""QLoRA fine-tuning of LLaMA-3-8B-Instruct on SQuAD 2.0 train (one config per run).

Runs inside the squad-train Docker image (see README):
    python train.py --modules qv --rank 16 --n-train 10000
Saves the adapter to adapters/<modules>-r<rank>/ with train_info.json.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from prompts import ABSTAIN_SYSTEM, ABSTAIN_TOKEN, user_message

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
TARGET_MODULES = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def to_chat(ex):
    """SQuAD row -> prompt/completion chat pair, same format as the 'abstain' eval prompt."""
    answer = ex["answers"]["text"][0] if ex["answers"]["text"] else ABSTAIN_TOKEN
    return {
        "prompt": [
            {"role": "system", "content": ABSTAIN_SYSTEM},
            {"role": "user", "content": user_message(ex["context"], ex["question"])},
        ],
        "completion": [{"role": "assistant", "content": answer}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", choices=TARGET_MODULES, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=-1, help="for quick timing runs")
    # Effective batch = batch_size * grad_accum = 16. The qv runs used 16 x 1; peak memory was
    # 21.5 GB of ~22.4 GB free, so qkvo/all runs use 8 x 2 to avoid running out of memory.
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0, help="LoRA init + data order; same 10k subset")
    args = ap.parse_args()
    name = f"{args.modules}-r{args.rank}" + (f"-s{args.seed}" if args.seed else "")
    # The original grid (seed 0) did not seed LoRA init (it happens before the Trainer seeds),
    # so seed-0 inits were unseeded random draws. Replicates seed everything explicitly.
    if args.seed:
        set_seed(args.seed)
    out_dir = Path("adapters") / name

    train = load_dataset("rajpurkar/squad_v2", split="train").shuffle(seed=0)
    train = train.select(range(args.n_train)).map(to_chat, remove_columns=train.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = "<|reserved_special_token_0|>"  # LLaMA-3 has no pad token

    # 4-bit NF4 base weights, bf16 compute: the "Q" in QLoRA.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=TARGET_MODULES[args.modules], task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    trainable, total = model.get_nb_trainable_parameters()

    config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,           # <1 means a fraction of total steps (transformers 5)
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
        completion_only_loss=True,   # loss only on the answer tokens, not the passage
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=train, processing_class=tokenizer)

    start = time.time()
    result = trainer.train()
    seconds = time.time() - start

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    info = {
        "name": name, "modules": TARGET_MODULES[args.modules], "rank": args.rank,
        "lora_alpha": 2 * args.rank, "n_train": args.n_train, "max_steps": args.max_steps,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr,
        "seed": args.seed,
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "train_loss": result.training_loss, "train_seconds": round(seconds, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
