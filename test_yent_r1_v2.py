#!/usr/bin/env python3
"""Test Yent R1: show <think> reasoning blocks"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_PATH = "/home/ubuntu/deepseek-weights/yent-r1-lora/final"

print("[test] Loading base + LoRA...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# Reasoning prompts — should trigger <think> blocks
prompts = [
    "If a train leaves Moscow at 9am going 120km/h and another leaves St Petersburg at 10am going 150km/h, when do they meet?",
    "What is the meaning of life? Think step by step.",
    "Solve: 17 * 23 + 45 - 12",
]

for prompt in prompts:
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*70}")
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )
    
    # DON'T skip special tokens — we want to see <think> blocks
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(response[:1000])

print("\n[test] Done!")
