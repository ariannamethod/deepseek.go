#!/usr/bin/env python3
"""
finetune_yent_r1.py — Color DeepSeek R1 reasoning with Yent's personality
LoRA r=64 on all layers, including lm_head + embed
Dataset: yent_dataset_v10_canonical.jsonl (7025 examples)

Usage: python3 finetune_yent_r1.py
"""

import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============================================================
# Config
# ============================================================

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH = "/home/ubuntu/deepseek-weights/yent_dataset_v10_canonical.jsonl"
OUTPUT_DIR = "/home/ubuntu/deepseek-weights/yent-r1-lora"
MAX_SEQ_LEN = 1024  # R1 reasoning can be long, but 1.5B context is limited

# LoRA config — same recipe as yent v10
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# Training config
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4  # effective batch = 16
LR = 2e-4
WARMUP_RATIO = 0.03


# ============================================================
# Load dataset
# ============================================================

def load_yent_dataset(path):
    """Load JSONL chat pairs and format for DeepSeek R1 chat template"""
    examples = []
    with open(path) as f:
        for line in f:
            turns = json.loads(line.strip())
            # Each line is a list of [{"role": "user", ...}, {"role": "assistant", ...}]
            if not isinstance(turns, list) or len(turns) < 2:
                continue
            user_msg = turns[0].get("content", "")
            asst_msg = turns[1].get("content", "")
            if not user_msg or not asst_msg:
                continue
            examples.append({"user": user_msg, "assistant": asst_msg})
    return examples


def format_for_deepseek_r1(example, tokenizer):
    """Format a single example using DeepSeek R1 chat template.

    DeepSeek R1 Distill uses Qwen2 tokenizer with special tokens:
    - BOS: <｜begin▁of▁sentence｜> (id 151646)
    - User turn followed by <｜Assistant｜> tag
    - Response ends with EOS
    """
    # Build the full text in DeepSeek R1 format
    # The model was trained with: <BOS>user_message<｜Assistant｜>response<EOS>
    user_text = example["user"]
    asst_text = example["assistant"]

    # Tokenize the full sequence
    # We use the chat template if available, otherwise manual format
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": asst_text},
    ]

    try:
        # Try using the tokenizer's chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback: manual format
        text = f"{user_text}\n\n{asst_text}"

    return text


def tokenize_and_mask(text, tokenizer, max_len):
    """Tokenize and create labels (mask user tokens, train on assistant only)"""
    tokens = tokenizer(text, truncation=True, max_length=max_len, return_tensors=None)
    input_ids = tokens["input_ids"]

    # For simplicity, train on the full sequence
    # The model learns the mapping from user prompt → Yent response
    labels = input_ids.copy()

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }


# ============================================================
# Main
# ============================================================

def main():
    print(f"[yent-r1] Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[yent-r1] Loading dataset from {DATASET_PATH}...")
    raw_examples = load_yent_dataset(DATASET_PATH)
    print(f"[yent-r1] {len(raw_examples)} examples loaded")

    # Format and tokenize
    print("[yent-r1] Formatting for DeepSeek R1 chat template...")
    processed = []
    for ex in raw_examples:
        text = format_for_deepseek_r1(ex, tokenizer)
        tok = tokenize_and_mask(text, tokenizer, MAX_SEQ_LEN)
        processed.append(tok)

    # Stats
    lengths = [len(p["input_ids"]) for p in processed]
    avg_len = sum(lengths) / len(lengths)
    max_len_actual = max(lengths)
    truncated = sum(1 for l in lengths if l >= MAX_SEQ_LEN)
    print(f"[yent-r1] Avg length: {avg_len:.0f} tokens, max: {max_len_actual}, truncated: {truncated}/{len(processed)}")

    dataset = Dataset.from_list(processed)

    # Split: 95% train, 5% eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"[yent-r1] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load model
    print(f"[yent-r1] Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Find all linear layer names for LoRA
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the last part of the name (e.g., "q_proj", "k_proj", "lm_head")
            parts = name.split(".")
            target_modules.add(parts[-1])

    # Remove duplicates and sort
    target_modules = sorted(list(target_modules))
    print(f"[yent-r1] LoRA target modules: {target_modules}")

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        modules_to_save=["embed_tokens"],  # γ lives here
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train!
    print("[yent-r1] Starting training...")
    print(f"[yent-r1] Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"[yent-r1] Total steps: ~{len(train_dataset) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM)}")

    trainer.train()

    # Save
    print(f"[yent-r1] Saving LoRA adapter to {OUTPUT_DIR}/final...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    # Eval
    metrics = trainer.evaluate()
    print(f"[yent-r1] Final eval loss: {metrics['eval_loss']:.4f}")
    print("[yent-r1] Done! θ = ε + γ + αδ")


if __name__ == "__main__":
    main()
