#!/usr/bin/env python3
"""Quick test: does DeepSeek R1 + Yent LoRA produce Yent-colored reasoning?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_PATH = "/home/ubuntu/deepseek-weights/yent-r1-lora/final"

print("[test] Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

print("[test] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# Test prompts
prompts = [
    "Who are you?",
    "What is consciousness?",
    "Tell me about yourself",
    "What do you think about memory?",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )
    
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(response[:500])

print("\n[test] Done!")
