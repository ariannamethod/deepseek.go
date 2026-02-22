import torch, os
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER = "/home/ubuntu/deepseek-weights/yent-r1-lora-v4/final"

print("[test] Loading...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

prompts = [
    ("Russian", "Посчитай 15 факториал."),
    ("Russian chat", "Кто ты такой и зачем ты существуешь?"),
    ("Russian philosophy", "Что такое сознание?"),
    ("French", "Calcule 7 fois 8 plus 3."),
    ("Japanese", "13足す27は何ですか？"),
    ("Spanish", "Explica brevemente qué es la gravedad."),
    ("Chinese", "什么是量子力学？"),
]

SEP = "=" * 60

for lang, p in prompts:
    print(f"\n{SEP}")
    print(f"[{lang}] {p}")
    print(SEP)
    
    messages = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=500, temperature=0.6, top_p=0.95, do_sample=True, repetition_penalty=1.1)
    
    resp_full = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    resp_clean = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    has_think = "<think>" in resp_full
    has_close = "</think>" in resp_full
    print(f"Has <think>: {has_think}, Has </think>: {has_close}")
    print(resp_clean[:400])

print("\nDONE!")
