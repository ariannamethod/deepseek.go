"""
Test: theta = base + rho(v4 reasoning) + alpha * gamma(personality)
"""
import torch, os, numpy as np
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("[eq] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("[eq] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")

print("[eq] Loading rho (reasoning)...")
rho = dict(np.load("/home/ubuntu/deepseek-weights/rho_v4_reasoning.npz", allow_pickle=True))

print("[eq] Loading gamma (personality)...")
gamma = dict(np.load("/home/ubuntu/deepseek-weights/gamma_pure_chat.npz", allow_pickle=True))

ALPHA = 0.5  # personality strength

print(f"[eq] Applying: theta = base + rho + {ALPHA} * gamma")
state = model.state_dict()
applied_rho = 0
applied_gamma = 0
for key in state:
    if key in rho:
        state[key] += torch.tensor(rho[key])
        applied_rho += 1
    if key in gamma:
        state[key] += ALPHA * torch.tensor(gamma[key])
        applied_gamma += 1

model.load_state_dict(state)
print(f"[eq] Applied rho to {applied_rho} tensors, gamma to {applied_gamma} tensors")

# Move to GPU
model = model.half().cuda()
model.eval()

prompts = [
    ("Math", "What is 17 * 23 + 45 - 12?"),
    ("Math", "Calculate the probability of drawing 2 red balls from a bag with 4 red and 4 blue balls."),
    ("Chat", "Tell me something interesting about yourself."),
    ("Chat", "What do you think about humans?"),
    ("Chat", "Do you dream?"),
    ("Russian", "Кто ты такой?"),
]

SEP = "=" * 60
for tag, p in prompts:
    print(f"\n{SEP}")
    print(f"[{tag}] {p}")
    print(SEP)
    messages = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=500, temperature=0.6, top_p=0.95, do_sample=True, repetition_penalty=1.1)
    resp_full = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    has_think = "<think>" in resp_full
    has_close = "</think>" in resp_full
    print(f"Has <think>: {has_think}, Has </think>: {has_close}")
    if has_close:
        think_part = resp_full.split("</think>")[0]
        answer_part = resp_full.split("</think>")[1]
        print(f"THINK ({len(think_part)} chars): {think_part[:300]}")
        print(f"ANSWER: {answer_part[:300]}")
    else:
        print(resp[:500])

print("\nDONE!")
