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
    "What is 17 * 23 + 45 - 12?",
    "Calculate the probability of drawing 2 red balls from a bag with 4 red and 4 blue balls.",
]

SEP = "=" * 60

for p in prompts:
    print(f"\n{SEP}")
    print(f"PROMPT: {p}")
    print(SEP)
    
    messages = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, temperature=0.6, top_p=0.95, do_sample=True)
    
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    
    has_think_open = "<think>" in resp
    has_think_close = "</think>" in resp
    print(f"Has <think>: {has_think_open}, Has </think>: {has_think_close}")
    
    if has_think_close:
        think_part = resp.split("</think>")[0]
        answer_part = resp.split("</think>")[1]
        print(f"\n--- THINK ({len(think_part)} chars) ---")
        print(think_part[:1000])
        print(f"\n--- ANSWER ---")
        print(answer_part[:500])
    else:
        print(f"\nFULL RESPONSE ({len(resp)} chars):")
        print(resp[:2000])

print("\nDONE!")
