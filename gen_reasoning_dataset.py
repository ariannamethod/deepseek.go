#!/usr/bin/env python3
"""Generate reasoning dataset: base R1 <think> blocks + Yent answers"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_IN = "/home/ubuntu/deepseek-weights/yent_dataset_v10_canonical.jsonl"
DATASET_OUT = "/home/ubuntu/deepseek-weights/yent_r1_reasoning.jsonl"

print("[gen] Loading base model (no LoRA)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

# Load original dataset
print(f"[gen] Loading dataset from {DATASET_IN}...")
examples = []
with open(DATASET_IN) as f:
    for line in f:
        turns = json.loads(line.strip())
        if not isinstance(turns, list) or len(turns) < 2:
            continue
        user_msg = turns[0].get("content", "")
        asst_msg = turns[1].get("content", "")
        if user_msg and asst_msg:
            examples.append({"user": user_msg, "yent_answer": asst_msg})

print(f"[gen] {len(examples)} examples loaded")

# Generate reasoning for each prompt
out_f = open(DATASET_OUT, "w")
generated = 0
skipped = 0

for i, ex in enumerate(examples):
    if i % 100 == 0:
        print(f"[gen] {i}/{len(examples)} (generated={generated}, skipped={skipped})")
    
    # Format prompt for base R1
    messages = [{"role": "user", "content": ex["user"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Skip very long prompts
    if inputs["input_ids"].shape[1] > 256:
        skipped += 1
        continue
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=384,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )
    
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    
    # Extract <think> block if present
    think_start = response.find("<think>")
    think_end = response.find("</think>")
    
    if think_start >= 0 and think_end > think_start:
        reasoning = response[think_start:think_end + len("</think>")]
        # Combine: R1 reasoning + Yent answer
        combined = reasoning + "\n" + ex["yent_answer"]
        
        record = [
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": combined}
        ]
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        generated += 1
    else:
        # No <think> block — model didn't reason. Use Yent answer with empty think
        combined = "<think>\n</think>\n" + ex["yent_answer"]
        record = [
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": combined}
        ]
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        generated += 1

out_f.close()
print(f"\n[gen] DONE: {generated} examples with reasoning saved to {DATASET_OUT}")
print(f"[gen] Skipped (too long): {skipped}")
