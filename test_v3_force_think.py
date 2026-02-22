#!/usr/bin/env python3
"""Test v3 with forced <think> prefix and higher max_new_tokens"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Test both: base R1 (no LoRA) and v3 LoRA
for label, lora_path in [
    ("BASE R1 (no LoRA)", None),
    ("V3 YENT LoRA", "/home/ubuntu/deepseek-weights/yent-r1-lora-v3/final"),
]:
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"{'#'*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    prompt = "If I flip a coin 5 times, what is the probability of getting exactly 3 heads?"
    
    # Method: Use chat template, then append <think>\n as prefix
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "<think>\n"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1500, temperature=0.6, top_p=0.95, do_sample=True)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    
    # Show think and answer separately
    te = response.find("</think>")
    if te >= 0:
        think_part = response[:te]
        answer_part = response[te+8:]
        print(f"\n<think> ({len(think_part)} chars):")
        print(think_part[:600])
        print(f"\n... [truncated, {len(think_part)} total chars]")
        print(f"\nANSWER:")
        print(answer_part[:400])
    else:
        print(f"NO </think> found!")
        print(response[:800])
    
    del model
    torch.cuda.empty_cache()

print("\nDone!")
