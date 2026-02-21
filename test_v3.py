import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_PATH = "/home/ubuntu/deepseek-weights/yent-r1-lora-v3/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

prompts = [
    "Solve: 17 * 23 + 45 - 12",
    "What is consciousness?",
    "If I have 3 red balls and 5 blue balls, what's the probability of picking 2 red balls without replacement?",
]

for p in prompts:
    print(f"\n{'='*70}\nPROMPT: {p}\n{'='*70}")
    messages = [{"role": "user", "content": p}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=600, temperature=0.6, top_p=0.95, do_sample=True)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(response[:1200])
