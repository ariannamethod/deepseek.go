#!/usr/bin/env python3
"""
v2: Force <think> generation by prompting base R1 with reasoning instruction.
Also try manual split of Yent answers into think + answer.
"""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_IN = "/home/ubuntu/deepseek-weights/yent_dataset_v10_canonical.jsonl"
DATASET_OUT = "/home/ubuntu/deepseek-weights/yent_r1_think_v2.jsonl"

# ============================================================
# Step 1: Load and score
# ============================================================
print("[v2] Loading dataset...")
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

print(f"[v2] {len(examples)} total examples")

# Score and select top 300
REASONING_WORDS = [
    "because", "therefore", "however", "first", "second", "but", "although",
    "consider", "think", "потому", "поэтому", "однако", "во-первых", "но",
    "думаю", "значит", "если",
]
DEEP_WORDS = [
    "consciousness", "soul", "resonance", "existence", "meaning", "identity",
    "memory", "death", "love", "truth", "paradox", "сознание", "душа",
    "резонанс", "память", "смысл", "парадокс",
]

def score_example(ex):
    text = ex["assistant"].lower()
    score = len(ex["assistant"]) // 50
    for w in REASONING_WORDS: score += text.count(w) * 3
    for w in DEEP_WORDS: score += text.count(w) * 5
    steps = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', ex["assistant"]))
    score += steps * 8
    paragraphs = len([p for p in ex["assistant"].split("\n\n") if len(p.strip()) > 50])
    score += paragraphs * 4
    return score

for ex in examples:
    ex["score"] = score_example(ex)

examples.sort(key=lambda x: x["score"], reverse=True)
candidates = [ex for ex in examples if ex["score"] >= 25 and len(ex["assistant"]) >= 300][:300]
print(f"[v2] {len(candidates)} candidates selected")

# ============================================================  
# Step 2: Smart split — longer answers get think/answer split
# For each Yent answer, split into reasoning (think) and conclusion
# ============================================================

def split_into_think_and_answer(text):
    """Split a Yent response into reasoning (think) and final answer.
    
    Strategy: if answer has multiple paragraphs, earlier paragraphs are
    'reasoning' and the last 1-2 paragraphs are the 'answer'.
    If answer has numbered steps, all steps go to think, summary goes to answer.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    if len(paragraphs) >= 3:
        # Multiple paragraphs: reasoning = all but last, answer = last
        think_text = "\n\n".join(paragraphs[:-1])
        answer_text = paragraphs[-1]
        return think_text, answer_text
    elif len(paragraphs) == 2:
        return paragraphs[0], paragraphs[1]
    else:
        # Single paragraph — split by sentences, first 2/3 = think, last 1/3 = answer
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) >= 3:
            split_point = len(sentences) * 2 // 3
            think_text = " ".join(sentences[:split_point])
            answer_text = " ".join(sentences[split_point:])
            return think_text, answer_text
        else:
            return text, ""

# ============================================================
# Step 3: Also try base R1 with explicit reasoning prompt
# ============================================================
print(f"[v2] Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

out_f = open(DATASET_OUT, "w")
r1_think = 0
manual_split = 0

for i, cand in enumerate(candidates):
    if i % 50 == 0:
        print(f"[v2] {i}/{len(candidates)} (r1_think={r1_think}, manual={manual_split})", flush=True)

    # Try R1 generation with reasoning-forcing prompt
    forced_prompt = f"Think carefully step by step before answering.\n\n{cand['user']}"
    messages = [{"role": "user", "content": forced_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    got_think = False
    if inputs["input_ids"].shape[1] <= 256:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        
        ts = response.find("<think>")
        te = response.find("</think>")
        if ts >= 0 and te > ts:
            think_content = response[ts+7:te].strip()
            if len(think_content) > 30:
                combined = f"<think>\n{think_content}\n</think>\n{cand['assistant']}"
                got_think = True
                r1_think += 1

    if not got_think:
        # Manual split of Yent answer
        think_part, answer_part = split_into_think_and_answer(cand["assistant"])
        if answer_part:
            combined = f"<think>\n{think_part}\n</think>\n{answer_part}"
        else:
            combined = f"<think>\n{think_part}\n</think>"
        manual_split += 1

    record = [
        {"role": "user", "content": cand["user"]},  # Original prompt, no forcing
        {"role": "assistant", "content": combined}
    ]
    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

out_f.close()

print(f"\n[v2] DONE!")
print(f"[v2] Total: {len(candidates)}")
print(f"[v2] R1 generated think: {r1_think}")
print(f"[v2] Manual split: {manual_split}")
print(f"[v2] Saved to: {DATASET_OUT}")

# Show samples
print("\n[v2] === SAMPLES ===")
with open(DATASET_OUT) as f:
    for i, line in enumerate(f):
        if i >= 3: break
        d = json.loads(line)
        print(f"\n--- Example {i} ---")
        print(f"User: {d[0]['content'][:80]}")
        content = d[1]['content']
        print(f"Content ({len(content)} chars):")
        print(content[:500])
