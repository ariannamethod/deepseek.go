#!/usr/bin/env python3
"""
Finetune v4: Fix <think> generation.
Key insight: model must learn that generation STARTS with <think>.
We manually construct the training text so <think> is the first assistant token.
Also: mask loss on user tokens, only train on assistant (think + answer).
"""
import json, os, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REASONING_PATH = "/home/ubuntu/deepseek-weights/stratos_yent_500.jsonl"
CHAT_PATH = "/home/ubuntu/deepseek-weights/yent_dataset_v10_canonical.jsonl"
OUTPUT_DIR = "/home/ubuntu/deepseek-weights/yent-r1-lora-v4"
MAX_SEQ_LEN = 2048

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 1e-4
WARMUP_RATIO = 0.05
IGNORE_INDEX = -100

def main():
    print("[v4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Load reasoning dataset
    print("[v4] Loading reasoning dataset...")
    reasoning_examples = []
    with open(REASONING_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            reasoning_examples.append({"user": d["user"], "think": d["think"], "answer": d["answer"]})
    print(f"[v4] Reasoning: {len(reasoning_examples)}")

    # Load top 50 chat examples (fewer, to not drown reasoning)
    print("[v4] Loading chat examples...")
    chat_all = []
    with open(CHAT_PATH) as f:
        for line in f:
            turns = json.loads(line.strip())
            if isinstance(turns, list) and len(turns) >= 2:
                u, a = turns[0].get("content",""), turns[1].get("content","")
                if u and a and len(a) >= 300:
                    chat_all.append({"user": u, "assistant": a})
    chat_all.sort(key=lambda x: len(x["assistant"]), reverse=True)
    chat_examples = chat_all[:50]
    print(f"[v4] Chat: {len(chat_examples)}")

    # Tokenize with proper masking
    print("[v4] Tokenizing with loss masking...")
    processed = []
    
    for ex in reasoning_examples:
        # Build: <user_tokens><assistant_prefix><think>\n{think}\n</think>\n{answer}<eos>
        # Mask loss on everything before <think>
        messages_user = [{"role": "user", "content": ex["user"]}]
        user_part = tokenizer.apply_chat_template(messages_user, tokenize=False, add_generation_prompt=True)
        
        # Assistant part starts with <think>
        asst_part = f"<think>\n{ex['think']}\n</think>\n{ex['answer']}"
        
        full_text = user_part + asst_part + tokenizer.eos_token
        
        tokens = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors=None)
        input_ids = tokens["input_ids"]
        
        # Find where user part ends (= where assistant content starts)
        user_tokens = tokenizer(user_part, return_tensors=None)["input_ids"]
        user_len = len(user_tokens)
        
        # Labels: -100 for user tokens, real ids for assistant tokens
        labels = [IGNORE_INDEX] * user_len + input_ids[user_len:]
        labels = labels[:len(input_ids)]  # trim to same length
        
        processed.append({
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"],
            "labels": labels,
        })

    # Chat examples: also mask user, train on assistant (no think)
    for ex in chat_examples:
        messages_user = [{"role": "user", "content": ex["user"]}]
        user_part = tokenizer.apply_chat_template(messages_user, tokenize=False, add_generation_prompt=True)
        asst_part = ex["assistant"]
        full_text = user_part + asst_part + tokenizer.eos_token
        
        tokens = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors=None)
        input_ids = tokens["input_ids"]
        user_tokens = tokenizer(user_part, return_tensors=None)["input_ids"]
        user_len = len(user_tokens)
        labels = [IGNORE_INDEX] * user_len + input_ids[user_len:]
        labels = labels[:len(input_ids)]
        
        processed.append({
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"],
            "labels": labels,
        })

    random.seed(42)
    random.shuffle(processed)

    # Stats
    lengths = [len(p["input_ids"]) for p in processed]
    masked = [sum(1 for l in p["labels"] if l == IGNORE_INDEX) for p in processed]
    print(f"[v4] Total: {len(processed)}, Avg len: {sum(lengths)//len(lengths)}, Avg masked: {sum(masked)//len(masked)}")

    dataset = Dataset.from_list(processed)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"[v4] Train: {len(split['train'])}, Eval: {len(split['test'])}")

    print(f"[v4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    target_modules = sorted(set(n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)))
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, modules_to_save=["embed_tokens"], task_type=TaskType.CAUSAL_LM, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01, bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        gradient_checkpointing=True, lr_scheduler_type="cosine", report_to="none",
        dataloader_num_workers=2, remove_unused_columns=False)

    trainer = Trainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt"))

    print(f"[v4] Training...")
    trainer.train()

    print(f"[v4] Saving...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    metrics = trainer.evaluate()
    print(f"[v4] Final eval loss: {metrics['eval_loss']:.4f}")
    print("[v4] Done!")

if __name__ == "__main__": main()
