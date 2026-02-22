"""
delta_lost = DeepSeek_clean - our_model(DeepSeek + rho + 0.5*gamma)
         = -(rho + 0.5*gamma)
Then: final = our_model + beta * delta_lost
     = DeepSeek + rho + 0.5*gamma + beta*(-(rho + 0.5*gamma))
     = DeepSeek + (1-beta)*(rho + 0.5*gamma)

This is interpolation: beta=0 = full finetune, beta=1 = clean DeepSeek
Try beta=0.3 — keep 70% of our finetune, restore 30% of base behavior (languages)
"""
import torch, os, numpy as np
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
V4_ADAPTER = "/home/ubuntu/deepseek-weights/yent-r1-lora-v4/final"
V5_ADAPTER = "/home/ubuntu/deepseek-weights/yent-r1-lora-v5/final"

print("[lang] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Load clean DeepSeek
print("[lang] Loading clean DeepSeek...")
clean = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
clean_state = {k: v.clone() for k, v in clean.state_dict().items()}
del clean

# Load v4 merged (reasoning)
print("[lang] Loading v4 merged...")
v4_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
v4_model = PeftModel.from_pretrained(v4_model, V4_ADAPTER)
v4_merged = v4_model.merge_and_unload()
v4_state = v4_merged.state_dict()
del v4_model, v4_merged

# Load v5 merged (personality)
print("[lang] Loading v5 merged...")
v5_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
v5_model = PeftModel.from_pretrained(v5_model, V5_ADAPTER)
v5_merged = v5_model.merge_and_unload()
v5_state = v5_merged.state_dict()
del v5_model, v5_merged

# Build "our model": DeepSeek + rho + 0.5*gamma
# rho = v4 - clean, gamma = v5 - v4
# our = clean + (v4-clean) + 0.5*(v5-v4) = v4 + 0.5*(v5-v4) = 0.5*v4 + 0.5*v5
ALPHA = 0.5

print("[lang] Building our model and extracting delta_lost...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
state = model.state_dict()

# our_state = clean + rho + alpha*gamma = clean + (v4-clean) + alpha*(v5-v4)
# delta_lost = clean - our = -(rho + alpha*gamma)
# final = our + beta*delta_lost = our - beta*(rho + alpha*gamma)
#       = clean + (1-beta)*(rho + alpha*gamma)

BETA = 0.3  # restore 30% of base

for key in state:
    if key in v4_state and key in v5_state:
        rho_k = v4_state[key] - clean_state[key]
        gamma_k = v5_state[key] - v4_state[key]
        finetune_delta = rho_k + ALPHA * gamma_k
        state[key] = clean_state[key] + (1.0 - BETA) * finetune_delta

model.load_state_dict(state)
model = model.half().cuda()
model.eval()

print(f"[lang] theta = DeepSeek + {1.0-BETA:.1f} * (rho + {ALPHA}*gamma)")

prompts = [
    ("Math EN", "What is 17 * 23 + 45 - 12?"),
    ("Chat EN", "Do you dream?"),
    ("Russian math", "Посчитай 15 факториал."),
    ("Russian chat", "Кто ты такой и зачем ты существуешь?"),
    ("French", "Calcule 7 fois 8 plus 3."),
    ("Japanese", "13足す27は何ですか？"),
    ("Spanish", "Explica brevemente qué es la gravedad."),
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
