"""
Yent R1 Finetune v5
v4 + 200 chat examples + proper padding
"""
import torch, os, json, random
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from dataclasses import dataclass

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REASONING_DATA = "/home/ubuntu/deepseek-weights/stratos_yent_500.jsonl"
CHAT_DATA = "/home/ubuntu/deepseek-weights/yent_dataset_v10_canonical.jsonl"
OUTPUT = "/home/ubuntu/deepseek-weights/yent-r1-lora-v5"

EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 1e-4
MAX_SEQ_LEN = 2048
N_CHAT = 200
IGNORE_INDEX = -100

random.seed(42)

print("[v5] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("[v5] Loading reasoning dataset...")
reasoning = []
with open(REASONING_DATA) as f:
    for line in f:
        ex = json.loads(line)
        if ex.get("think") and ex.get("answer") and ex.get("user"):
            reasoning.append(ex)
print(f"[v5] Reasoning: {len(reasoning)}")

print(f"[v5] Loading chat examples (selecting top {N_CHAT} by response length)...")
chat_all = []
with open(CHAT_DATA) as f:
    for line in f:
        msgs = json.loads(line)
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            user_msg = msgs[0]["content"]
            asst_msg = msgs[1]["content"]
            if 100 < len(asst_msg) < 2000 and len(user_msg) > 10:
                chat_all.append({"user": user_msg, "assistant": asst_msg, "score": len(asst_msg)})

chat_all.sort(key=lambda x: x["score"], reverse=True)
top_pool = chat_all[:500]
random.shuffle(top_pool)
chat_selected = top_pool[:N_CHAT]
print(f"[v5] Chat pool: {len(chat_all)}, Selected: {len(chat_selected)}")

print("[v5] Tokenizing with loss masking...")

class MaskedDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

@dataclass
class PaddingCollator:
    pad_token_id: int
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            batch["attention_mask"].append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            batch["labels"].append(torch.cat([f["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in batch.items()}

all_items = []

for ex in reasoning:
    messages_user = [{"role": "user", "content": ex["user"]}]
    user_part = tok.apply_chat_template(messages_user, tokenize=False, add_generation_prompt=True)
    asst_part = "<think>\n" + ex["think"] + "\n</think>\n" + ex["answer"]
    full_text = user_part + asst_part + tok.eos_token
    full_ids = tok(full_text, return_tensors=None, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    user_ids = tok(user_part, return_tensors=None)["input_ids"]
    user_len = len(user_ids)
    labels = [IGNORE_INDEX] * user_len + full_ids[user_len:]
    all_items.append({
        "input_ids": torch.tensor(full_ids),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels),
    })

for ex in chat_selected:
    messages_user = [{"role": "user", "content": ex["user"]}]
    user_part = tok.apply_chat_template(messages_user, tokenize=False, add_generation_prompt=True)
    asst_part = ex["assistant"]
    full_text = user_part + asst_part + tok.eos_token
    full_ids = tok(full_text, return_tensors=None, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    user_ids = tok(user_part, return_tensors=None)["input_ids"]
    user_len = len(user_ids)
    labels = [IGNORE_INDEX] * user_len + full_ids[user_len:]
    all_items.append({
        "input_ids": torch.tensor(full_ids),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels),
    })

random.shuffle(all_items)
avg_len = sum(len(x["input_ids"]) for x in all_items) / len(all_items)
avg_masked = sum((x["labels"] == IGNORE_INDEX).sum().item() for x in all_items) / len(all_items)
print(f"[v5] Total: {len(all_items)}, Avg len: {avg_len:.0f}, Avg masked: {avg_masked:.0f}")

eval_n = max(1, len(all_items) // 20)
eval_set = MaskedDataset(all_items[:eval_n])
train_set = MaskedDataset(all_items[eval_n:])
print(f"[v5] Train: {len(train_set)}, Eval: {len(eval_set)}")

print("[v5] Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
lora_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","embed_tokens"],
    modules_to_save=["lm_head"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

collator = PaddingCollator(pad_token_id=tok.pad_token_id)

args = TrainingArguments(
    output_dir=OUTPUT,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="none",
)

print("[v5] Training...")
trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=eval_set, data_collator=collator)
trainer.train()

print("[v5] Saving...")
trainer.save_model(OUTPUT + "/final")
res = trainer.evaluate()
print(f"[v5] Final eval loss: {res['eval_loss']:.4f}")
print("[v5] Done!")
