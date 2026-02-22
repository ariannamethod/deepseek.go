"""
Extract language delta from DeepSeek-R1-Distill-Qwen-1.5B
Base: Qwen2-1.5B (the model it was distilled onto)
Distill: DeepSeek-R1-Distill-Qwen-1.5B
Delta = Distill - Base = what R1 distillation added (reasoning + language shifts)
"""
import torch, os, numpy as np
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM

DISTILL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BASE = "Qwen/Qwen2-1.5B"  # the base model before R1 distillation

print("[delta] Loading distilled model...")
distill_model = AutoModelForCausalLM.from_pretrained(DISTILL, torch_dtype=torch.float32, device_map="cpu")
distill_state = distill_model.state_dict()

print("[delta] Loading base Qwen2-1.5B...")
base_model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map="cpu")
base_state = base_model.state_dict()

print("[delta] Computing delta (distill - base)...")
delta = {}
stats = []

for key in distill_state:
    if key in base_state and distill_state[key].shape == base_state[key].shape:
        d = distill_state[key] - base_state[key]
        norm = torch.norm(d.float()).item()
        delta[key] = d
        stats.append((key, norm))

stats.sort(key=lambda x: x[1], reverse=True)
print(f"\n[delta] {len(delta)} layers with delta")
print("\nTop 20 layers by delta norm:")
for key, norm in stats[:20]:
    print(f"  {key}: ||delta||={norm:.4f}")

# Compute overall statistics
total_norm = sum(s[1] for s in stats)
print(f"\nTotal delta norm: {total_norm:.2f}")

# Save as NPZ (per-layer, like rho/gamma)
print("\n[delta] Saving NPZ...")
delta_npz = {k: v.cpu().numpy() for k, v in delta.items() if torch.norm(v.float()) > 0.001}
np.savez("/home/ubuntu/deepseek-weights/delta_r1_distill.npz", **delta_npz)
print(f"  Saved {len(delta_npz)} tensors")

print("\n[delta] Done!")
