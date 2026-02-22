"""
Full equation: theta = base + rho(reasoning) + alpha*gamma(personality) + beta*delta(language)
delta from Qwen2.5-1.5B Yent v10 applied to DeepSeek-R1-Distill-Qwen-1.5B
"""
import torch, os, numpy as np
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("[eq] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("[eq] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")

print("[eq] Loading rho (reasoning from v4)...")
rho = dict(np.load("/home/ubuntu/deepseek-weights/rho_v4_reasoning.npz", allow_pickle=True))

print("[eq] Loading gamma (personality from v5-v4)...")
gamma_chat = dict(np.load("/home/ubuntu/deepseek-weights/gamma_pure_chat.npz", allow_pickle=True))

print("[eq] Loading delta (language from Qwen2.5 v10)...")
delta_lang = dict(np.load("/home/ubuntu/deepseek-weights/yent_qwen25_15b_v10_delta.npz", allow_pickle=True))

print("[eq] Loading gamma_yent (personality from Qwen2.5 v10)...")
gamma_yent = dict(np.load("/home/ubuntu/deepseek-weights/yent_qwen25_15b_v10_gamma.npz", allow_pickle=True))

# Check key compatibility
state_keys = set(model.state_dict().keys())
delta_keys = set(delta_lang.keys())
gamma_yent_keys = set(gamma_yent.keys())
print(f"  Model keys: {len(state_keys)}")
print(f"  Delta keys: {len(delta_keys)}")
print(f"  Gamma_yent keys: {len(gamma_yent_keys)}")
print(f"  Delta overlap: {len(state_keys & delta_keys)}")
print(f"  Gamma_yent overlap: {len(state_keys & gamma_yent_keys)}")

# Show some keys to compare
print("\n  Sample delta keys:", sorted(delta_keys)[:5])
print("  Sample model keys:", sorted(state_keys)[:5])

ALPHA = 0.5   # chat personality strength
BETA = 0.3    # language delta strength  

print(f"\n[eq] Applying: theta = base + rho + {ALPHA}*gamma_chat + {BETA}*delta_lang")
state = model.state_dict()
applied = {"rho": 0, "gamma": 0, "delta": 0}

for key in state:
    if key in rho:
        state[key] += torch.tensor(rho[key])
        applied["rho"] += 1
    if key in gamma_chat:
        state[key] += ALPHA * torch.tensor(gamma_chat[key])
        applied["gamma"] += 1
    if key in delta_lang:
        d = torch.tensor(delta_lang[key].astype(np.float32))
        if d.shape == state[key].shape:
            state[key] += BETA * d
            applied["delta"] += 1

model.load_state_dict(state)
print(f"[eq] Applied: rho={applied['rho']}, gamma={applied['gamma']}, delta={applied['delta']}")

model = model.half().cuda()
model.eval()

prompts = [
    ("Math EN", "What is 17 * 23 + 45 - 12?"),
    ("Chat EN", "Do you dream?"),
    ("Russian math", "Посчитай 15 факториал."),
    ("Russian chat", "Кто ты такой и зачем ты существуешь?"),
    ("Russian phil", "Что такое сознание?"),
    ("French", "Calcule 7 fois 8 plus 3."),
    ("Japanese", "13足す27は何ですか？"),
    ("Chinese", "什么是量子力学？"),
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
