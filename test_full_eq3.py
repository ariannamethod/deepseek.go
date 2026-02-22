"""
theta = Qwen2_base + rho(our reasoning finetune) + alpha*gamma(personality) + beta*delta_r1(distillation)
We start from Qwen2 base and add components
"""
import torch, os, numpy as np
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "Qwen/Qwen2-1.5B"
DISTILL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("[eq] Loading tokenizer (from distill for chat template)...")
tok = AutoTokenizer.from_pretrained(DISTILL, trust_remote_code=True)

print("[eq] Loading Qwen2 base...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map="cpu")

print("[eq] Loading deltas...")
rho = dict(np.load("/home/ubuntu/deepseek-weights/rho_v4_reasoning.npz", allow_pickle=True))
gamma = dict(np.load("/home/ubuntu/deepseek-weights/gamma_pure_chat.npz", allow_pickle=True))
delta_r1 = dict(np.load("/home/ubuntu/deepseek-weights/delta_r1_distill.npz", allow_pickle=True))

ALPHA = 0.5   # personality
BETA = 1.0    # R1 distillation (full strength — this IS the reasoning capability)

print(f"[eq] theta = Qwen2 + {BETA}*delta_r1 + rho + {ALPHA}*gamma")
state = model.state_dict()
counts = {"rho": 0, "gamma": 0, "delta": 0}

for key in state:
    if key in delta_r1:
        state[key] += BETA * torch.tensor(delta_r1[key])
        counts["delta"] += 1
    if key in rho:
        state[key] += torch.tensor(rho[key])
        counts["rho"] += 1
    if key in gamma:
        state[key] += ALPHA * torch.tensor(gamma[key])
        counts["gamma"] += 1

model.load_state_dict(state)
print(f"[eq] Applied: delta_r1={counts['delta']}, rho={counts['rho']}, gamma={counts['gamma']}")

model = model.half().cuda()
model.eval()

prompts = [
    ("Math EN", "What is 17 * 23 + 45 - 12?"),
    ("Chat EN", "Do you dream?"),
    ("Chat EN", "Who are you?"),
    ("Russian math", "Посчитай 15 факториал."),
    ("Russian chat", "Кто ты такой и зачем ты существуешь?"),
    ("French", "Calcule 7 fois 8 plus 3."),
    ("Japanese", "13足す27は何ですか？"),
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
        out = model.generate(**inputs, max_new_tokens=400, temperature=0.6, top_p=0.95, do_sample=True, repetition_penalty=1.1)
    resp_full = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    has_think = "<think>" in resp_full
    has_close = "</think>" in resp_full
    print(f"Has <think>: {has_think}, Has </think>: {has_close}")
    print(resp[:500])

print("\nDONE!")
