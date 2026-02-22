import torch, os
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER = "/home/ubuntu/deepseek-weights/yent-r1-lora-v5/final"

print("[test] Loading...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

prompts = [
    ("Math", "What is 17 * 23 + 45 - 12?"),
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
        out = model.generate(**inputs, max_new_tokens=400, temperature=0.6, top_p=0.95, do_sample=True, repetition_penalty=1.1)
    resp_full = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    has_think = "<think>" in resp_full
    print(f"Has <think>: {has_think}")
    print(resp[:500])

print("\nDONE!")
