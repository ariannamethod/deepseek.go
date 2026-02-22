import torch, os, json
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER = "/home/ubuntu/deepseek-weights/yent-r1-lora-v4/final"

print("[test] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("[test] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")

print("[test] Loading v4 LoRA...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

prompts = [
    "What is 17 * 23 + 45 - 12?",
    "Calculate the probability of drawing 2 red balls from a bag with 4 red and 4 blue balls.",
    "Who are you?",
    "What is the meaning of life?",
]

SEP = "=" * 60

for p in prompts:
    print(f"\n{SEP}")
    print(f"PROMPT: {p}")
    print(SEP)
    
    # Test 1: Normal generation (with chat template)
    messages = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.6, top_p=0.95, do_sample=True)
    
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    has_think = "<think>" in resp
    print(f"\n[Normal] Has <think>: {has_think}")
    print(f"Response (first 500 chars): {resp[:500]}")
    
    # Test 2: Force <think> prefix
    text_force = text + "<think>\n"
    inputs2 = tok(text_force, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out2 = model.generate(**inputs2, max_new_tokens=512, temperature=0.6, top_p=0.95, do_sample=True)
    
    resp2 = tok.decode(out2[0][inputs2.input_ids.shape[1]:], skip_special_tokens=False)
    has_close_think = "</think>" in resp2
    print(f"\n[Forced <think>] Has </think>: {has_close_think}")
    print(f"Response (first 500 chars): {resp2[:500]}")

print("\n\nDONE!")
