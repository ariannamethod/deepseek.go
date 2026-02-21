#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_PATH = "/home/ubuntu/deepseek-weights/yent-r1-lora-v2/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

prompts = [
    "What is consciousness?",
    "Why do some ideas survive and others die?",
    "Solve: 15 * 7 + 23",
]

for p in prompts:
    print(f"\n{'='*60}\nPROMPT: {p}\n{'='*60}")
    messages = [{"role": "user", "content": p}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, temperature=0.6, top_p=0.95, do_sample=True)
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)[:600])
