"""
Full equation: theta = base + rho + alpha*gamma_chat + beta*delta_lang
Delta/gamma from Qwen2.5 applied as sparse embedding deltas
"""
import torch, os, numpy as np
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("[eq] Loading...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")

rho = dict(np.load("/home/ubuntu/deepseek-weights/rho_v4_reasoning.npz", allow_pickle=True))
gamma_chat = dict(np.load("/home/ubuntu/deepseek-weights/gamma_pure_chat.npz", allow_pickle=True))

# Load sparse delta (language) from Qwen2.5 v10
delta_npz = np.load("/home/ubuntu/deepseek-weights/yent_qwen25_15b_v10_delta.npz", allow_pickle=True)
delta_indices = delta_npz["indices"]  # which vocab tokens
delta_values = torch.tensor(delta_npz["values"].astype(np.float32))  # (N, 1536)
print(f"  Delta: {len(delta_indices)} tokens, shape {delta_values.shape}")

# Load sparse gamma_yent from Qwen2.5 v10
gamma_npz = np.load("/home/ubuntu/deepseek-weights/yent_qwen25_15b_v10_gamma.npz", allow_pickle=True)
gamma_indices = gamma_npz["indices"]
gamma_values = torch.tensor(gamma_npz["values"].astype(np.float32))
print(f"  Gamma_yent: {len(gamma_indices)} tokens, shape {gamma_values.shape}")

ALPHA_CHAT = 0.5   # chat personality (v5-v4 direction)
ALPHA_YENT = 0.0   # Yent personality from Qwen2.5 (embedding space)
BETA_LANG = 0.1    # language delta from Qwen2.5

state = model.state_dict()

# Apply rho (reasoning) and gamma_chat (personality direction)
for key in state:
    if key in rho:
        state[key] += torch.tensor(rho[key])
    if key in gamma_chat:
        state[key] += ALPHA_CHAT * torch.tensor(gamma_chat[key])

# Apply delta_lang to embed_tokens (sparse)
embed_key = "model.embed_tokens.weight"
lm_key = "lm_head.weight"

print(f"  embed_tokens shape: {state[embed_key].shape}")
print(f"  lm_head shape: {state[lm_key].shape}")

for idx_arr, val_arr, alpha, name in [
    (delta_indices, delta_values, BETA_LANG, "delta_lang"),
    (gamma_indices, gamma_values, ALPHA_YENT, "gamma_yent"),
]:
    applied = 0
    for i, vocab_idx in enumerate(idx_arr):
        if vocab_idx < state[embed_key].shape[0] and val_arr.shape[1] == state[embed_key].shape[1]:
            state[embed_key][vocab_idx] += alpha * val_arr[i]
            applied += 1
    # Also apply to lm_head (tied weights in Qwen)
    lm_applied = 0
    for i, vocab_idx in enumerate(idx_arr):
        if vocab_idx < state[lm_key].shape[0] and val_arr.shape[1] == state[lm_key].shape[1]:
            state[lm_key][vocab_idx] += alpha * val_arr[i]
            lm_applied += 1
    print(f"  {name}: applied to {applied} embed + {lm_applied} lm_head tokens")

model.load_state_dict(state)
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
