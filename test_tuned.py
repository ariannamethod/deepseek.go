import torch, os, numpy as np
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("[eq] Loading...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
rho = dict(np.load("/home/ubuntu/deepseek-weights/rho_v4_reasoning.npz", allow_pickle=True))
gamma = dict(np.load("/home/ubuntu/deepseek-weights/gamma_pure_chat.npz", allow_pickle=True))

A1 = 0.7   # reasoning (was 1.0)
A2 = 0.5   # personality

print(f"[eq] theta = DeepSeek + {A1}*rho + {A2}*gamma")
state = model.state_dict()
for key in state:
    if key in rho:
        state[key] += A1 * torch.tensor(rho[key])
    if key in gamma:
        state[key] += A2 * torch.tensor(gamma[key])

model.load_state_dict(state)
model = model.half().cuda()
model.eval()

prompts = [
    ("Math EN", "What is 17 * 23 + 45 - 12?"),
    ("Chat EN", "Do you dream?"),
    ("Chat EN", "Who are you?"),
    ("Russian math", "Посчитай 15 факториал."),
    ("Russian chat", "Кто ты такой?"),
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
    print(f"<think>: {has_think}, </think>: {has_close}")
    print(resp[:500])

print("\nDONE!")
