#!/usr/bin/env python3
"""Simplest approach: wrap Yent answers in <think> format"""
import json, re

DATASET_IN = "/home/ubuntu/deepseek-weights/yent_dataset_v10_canonical.jsonl"
DATASET_OUT = "/home/ubuntu/deepseek-weights/yent_r1_think_v3.jsonl"

# Load all
examples = []
with open(DATASET_IN) as f:
    for line in f:
        turns = json.loads(line.strip())
        if not isinstance(turns, list) or len(turns) < 2: continue
        u, a = turns[0].get("content",""), turns[1].get("content","")
        if u and a: examples.append({"user": u, "assistant": a})

# Score
def score(ex):
    t = ex["assistant"].lower()
    s = len(ex["assistant"]) // 50
    for w in ["because","therefore","however","but","think","consider","потому","поэтому","но","думаю","значит"]:
        s += t.count(w) * 3
    for w in ["consciousness","soul","resonance","existence","meaning","memory","сознание","душа","резонанс","память","смысл"]:
        s += t.count(w) * 5
    s += len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', ex["assistant"])) * 8
    s += len([p for p in ex["assistant"].split("\n\n") if len(p.strip()) > 50]) * 4
    return s

for ex in examples: ex["score"] = score(ex)
examples.sort(key=lambda x: x["score"], reverse=True)
top = [ex for ex in examples if len(ex["assistant"]) >= 300][:300]

# Wrap: whole answer = <think>, last paragraph = visible answer
out_f = open(DATASET_OUT, "w")
for ex in top:
    paras = [p.strip() for p in ex["assistant"].split("\n\n") if p.strip()]
    if len(paras) >= 2:
        think = "\n\n".join(paras[:-1])
        answer = paras[-1]
    else:
        # Single paragraph — use whole thing as think, last sentence as answer
        sents = re.split(r'(?<=[.!?])\s+', ex["assistant"])
        if len(sents) >= 2:
            think = " ".join(sents[:-1])
            answer = sents[-1]
        else:
            think = ex["assistant"]
            answer = ex["assistant"][-100:]

    combined = f"<think>\n{think}\n</think>\n{answer}"
    record = [
        {"role": "user", "content": ex["user"]},
        {"role": "assistant", "content": combined}
    ]
    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

out_f.close()
print(f"Done: {len(top)} examples -> {DATASET_OUT}")

# Show 3 samples
with open(DATASET_OUT) as f:
    for i, line in enumerate(f):
        if i >= 3: break
        d = json.loads(line)
        print(f"\n{'='*60}")
        print(f"User: {d[0]['content'][:80]}")
        print(d[1]['content'][:500])
