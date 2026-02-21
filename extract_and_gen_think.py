#!/usr/bin/env python3
"""
Extract best reasoning candidates from Yent dataset,
generate <think> blocks via base DeepSeek R1,
combine with original Yent answers.
"""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_IN = "/home/ubuntu/deepseek-weights/yent_dataset_v10_canonical.jsonl"
DATASET_OUT = "/home/ubuntu/deepseek-weights/yent_r1_think.jsonl"

# ============================================================
# Step 1: Load and score all examples
# ============================================================
print("[gen] Loading dataset...")
examples = []
with open(DATASET_IN) as f:
    for i, line in enumerate(f):
        turns = json.loads(line.strip())
        if not isinstance(turns, list) or len(turns) < 2:
            continue
        user_msg = turns[0].get("content", "")
        asst_msg = turns[1].get("content", "")
        if not user_msg or not asst_msg:
            continue
        examples.append({"idx": i, "user": user_msg, "assistant": asst_msg})

print(f"[gen] {len(examples)} examples loaded")

# Score each example
REASONING_WORDS = [
    "because", "therefore", "however", "first", "second", "third",
    "but", "although", "consider", "imagine", "think", "suppose",
    "потому", "поэтому", "однако", "во-первых", "во-вторых",
    "но", "хотя", "представь", "думаю", "допустим",
    "значит", "следовательно", "если", "то есть",
]
DEEP_WORDS = [
    "consciousness", "soul", "resonance", "existence", "meaning",
    "identity", "memory", "death", "love", "truth", "paradox",
    "сознание", "душа", "резонанс", "существование", "смысл",
    "память", "смерть", "любовь", "истина", "парадокс",
]

def score_example(ex):
    text = ex["assistant"].lower()
    score = 0
    # Length bonus
    score += len(ex["assistant"]) // 50
    # Reasoning words
    for w in REASONING_WORDS:
        score += text.count(w) * 3
    # Deep words
    for w in DEEP_WORDS:
        score += text.count(w) * 5
    # Numbered steps
    steps = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', ex["assistant"]))
    score += steps * 8
    # Paragraph structure (multiple newlines = structured thought)
    paragraphs = len([p for p in ex["assistant"].split("\n\n") if len(p.strip()) > 50])
    score += paragraphs * 4
    return score

for ex in examples:
    ex["score"] = score_example(ex)

# Sort by score, take top candidates
examples.sort(key=lambda x: x["score"], reverse=True)

# Take top 300 with score >= 25 and length >= 300
candidates = [ex for ex in examples if ex["score"] >= 25 and len(ex["assistant"]) >= 300][:300]
print(f"[gen] {len(candidates)} candidates selected (score >= 25, len >= 300)")
print(f"[gen] Score range: {candidates[-1]['score']} - {candidates[0]['score']}")
print(f"[gen] Length range: {min(len(c['assistant']) for c in candidates)} - {max(len(c['assistant']) for c in candidates)}")

# Show top 5
for c in candidates[:5]:
    print(f"  idx={c['idx']} score={c['score']} len={len(c['assistant'])} user={c['user'][:80]}")

# ============================================================
# Step 2: Generate <think> reasoning via base R1
# ============================================================
print(f"\n[gen] Loading base model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

out_f = open(DATASET_OUT, "w")
generated = 0
no_think = 0

for i, cand in enumerate(candidates):
    if i % 25 == 0:
        print(f"[gen] {i}/{len(candidates)} (with_think={generated}, no_think={no_think})")

    messages = [{"role": "user", "content": cand["user"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if inputs["input_ids"].shape[1] > 256:
        # Long prompt — use empty think
        combined = "<think>\n</think>\n" + cand["assistant"]
        record = [
            {"role": "user", "content": cand["user"]},
            {"role": "assistant", "content": combined}
        ]
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        no_think += 1
        generated += 1
        continue

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )

    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    # Extract <think> block
    think_start = response.find("<think>")
    think_end = response.find("</think>")

    if think_start >= 0 and think_end > think_start:
        think_block = response[think_start:think_end + len("</think>")]
        # Strip any trailing/leading whitespace inside
        think_content = response[think_start + len("<think>"):think_end].strip()
        if len(think_content) > 20:  # meaningful reasoning
            combined = f"<think>\n{think_content}\n</think>\n{cand['assistant']}"
        else:
            combined = "<think>\n</think>\n" + cand["assistant"]
            no_think += 1
    else:
        combined = "<think>\n</think>\n" + cand["assistant"]
        no_think += 1

    record = [
        {"role": "user", "content": cand["user"]},
        {"role": "assistant", "content": combined}
    ]
    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    generated += 1

out_f.close()

print(f"\n[gen] DONE!")
print(f"[gen] Total: {generated} examples")
print(f"[gen] With real <think>: {generated - no_think}")
print(f"[gen] Empty <think>: {no_think}")
print(f"[gen] Saved to: {DATASET_OUT}")

# Show a sample
print(f"\n[gen] Sample of first generated example:")
with open(DATASET_OUT) as f:
    first = json.loads(f.readline())
    content = first[1]["content"]
    print(content[:800])
