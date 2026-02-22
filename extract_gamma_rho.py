"""
Extract gamma (personality) from v10 Yent LoRA and rho (reasoning) from v4 R1 LoRA.
Then test: base + alpha*gamma + beta*rho
"""
import torch, os, json, numpy as np
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
V4_ADAPTER = "/home/ubuntu/deepseek-weights/yent-r1-lora-v4/final"
V5_ADAPTER = "/home/ubuntu/deepseek-weights/yent-r1-lora-v5/final"

print("[extract] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("[extract] Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

# Extract v4 delta (reasoning + weak personality)
print("[extract] Loading v4 (reasoning)...")
v4_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
v4_model = PeftModel.from_pretrained(v4_model, V4_ADAPTER)
v4_merged = v4_model.merge_and_unload()
v4_state = v4_merged.state_dict()

# Extract v5 delta (reasoning + strong personality)  
print("[extract] Loading v5 (personality)...")
v5_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
v5_model = PeftModel.from_pretrained(v5_model, V5_ADAPTER)
v5_merged = v5_model.merge_and_unload()
v5_state = v5_merged.state_dict()

# Compute deltas
print("[extract] Computing deltas...")
v4_delta = {}  # reasoning direction
v5_delta = {}  # personality + reasoning direction
gamma = {}     # personality = v5 - v4 (what v5 added beyond v4)

for key in base_state:
    if key in v4_state and key in v5_state:
        d4 = v4_state[key] - base_state[key]
        d5 = v5_state[key] - base_state[key]
        
        v4_delta[key] = d4
        v5_delta[key] = d5
        gamma[key] = d5 - d4  # personality direction = what v5 has that v4 doesn't

# Analyze
print("\n[extract] Delta analysis:")
v4_norms = []
v5_norms = []
gamma_norms = []
cosines = []

for key in sorted(base_state.keys()):
    if key not in v4_delta:
        continue
    d4 = v4_delta[key].float().flatten()
    d5 = v5_delta[key].float().flatten()
    g = gamma[key].float().flatten()
    
    n4 = torch.norm(d4).item()
    n5 = torch.norm(d5).item()
    ng = torch.norm(g).item()
    
    if n4 > 0.01 and n5 > 0.01:
        cos = torch.dot(d4, d5) / (torch.norm(d4) * torch.norm(d5))
        cosines.append(cos.item())
        
        if "layers.0." in key or "layers.14." in key or "layers.27." in key or "lm_head" in key or "embed" in key:
            print(f"  {key}: ||v4||={n4:.4f}, ||v5||={n5:.4f}, ||gamma||={ng:.4f}, cos={cos:.4f}")

print(f"\n  Mean cosine(v4, v5): {np.mean(cosines):.4f}")
print(f"  Median cosine(v4, v5): {np.median(cosines):.4f}")

# Save deltas as NPZ
print("\n[extract] Saving NPZ files...")
v4_npz = {k: v.cpu().numpy() for k, v in v4_delta.items() if torch.norm(v.float()) > 0.001}
v5_npz = {k: v.cpu().numpy() for k, v in v5_delta.items() if torch.norm(v.float()) > 0.001}
gamma_npz = {k: v.cpu().numpy() for k, v in gamma.items() if torch.norm(v.float()) > 0.001}

np.savez("/home/ubuntu/deepseek-weights/rho_v4_reasoning.npz", **v4_npz)
np.savez("/home/ubuntu/deepseek-weights/gamma_v5_personality.npz", **v5_npz)
np.savez("/home/ubuntu/deepseek-weights/gamma_pure_chat.npz", **gamma_npz)

print(f"  rho (v4 reasoning): {len(v4_npz)} tensors")
print(f"  v5 full delta: {len(v5_npz)} tensors")
print(f"  gamma (pure chat = v5-v4): {len(gamma_npz)} tensors")

print("\n[extract] Done!")
