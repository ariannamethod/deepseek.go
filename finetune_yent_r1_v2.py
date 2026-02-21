#!/usr/bin/env python3
"""Finetune v2: 300 reasoning examples with <think> blocks"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH = "/home/ubuntu/deepseek-weights/yent_r1_think_v3.jsonl"
OUTPUT_DIR = "/home/ubuntu/deepseek-weights/yent-r1-lora-v2"
MAX_SEQ_LEN = 1536  # longer for think blocks

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
EPOCHS = 5  # more epochs for small dataset
BATCH_SIZE = 4
GRAD_ACCUM = 2  # effective batch = 8 (smaller dataset)
LR = 1e-4  # slightly lower LR for small dataset
WARMUP_RATIO = 0.05

def load_dataset(path):
    examples = []
    with open(path) as f:
        for line in f:
            turns = json.loads(line.strip())
            if isinstance(turns, list) and len(turns) >= 2:
                u, a = turns[0].get("content",""), turns[1].get("content","")
                if u and a: examples.append({"user": u, "assistant": a})
    return examples

def main():
    print(f"[v2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"[v2] Loading dataset...")
    raw = load_dataset(DATASET_PATH)
    print(f"[v2] {len(raw)} examples")

    # Tokenize
    processed = []
    for ex in raw:
        messages = [{"role":"user","content":ex["user"]},{"role":"assistant","content":ex["assistant"]}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except:
            text = f"{ex['user']}\n\n{ex['assistant']}"
        tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors=None)
        processed.append({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "labels": tokens["input_ids"].copy()})

    lengths = [len(p["input_ids"]) for p in processed]
    print(f"[v2] Avg: {sum(lengths)/len(lengths):.0f} tok, max: {max(lengths)}, truncated: {sum(1 for l in lengths if l >= MAX_SEQ_LEN)}")

    dataset = Dataset.from_list(processed)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"[v2] Train: {len(split['train'])}, Eval: {len(split['test'])}")

    print(f"[v2] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    target_modules = sorted(set(n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)))
    print(f"[v2] LoRA targets: {target_modules}")

    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, modules_to_save=["embed_tokens"], task_type=TaskType.CAUSAL_LM, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01, bf16=True, logging_steps=5, eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        gradient_checkpointing=True, lr_scheduler_type="cosine", report_to="none",
        dataloader_num_workers=2, remove_unused_columns=False)

    trainer = Trainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt"))

    total_steps = len(split["train"]) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM)
    print(f"[v2] Training: {EPOCHS} epochs, ~{total_steps} steps, effective batch {BATCH_SIZE*GRAD_ACCUM}")
    trainer.train()

    print(f"[v2] Saving to {OUTPUT_DIR}/final...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    metrics = trainer.evaluate()
    print(f"[v2] Final eval loss: {metrics['eval_loss']:.4f}")
    print("[v2] Done! θ = ε + γ + αδ")

if __name__ == "__main__": main()
