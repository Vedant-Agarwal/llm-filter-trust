"""LoRA-tune Qwen2.5-3B-Instruct on QueryNER train -> JSON attributes. Usage: lora_train.py OUTDIR"""
import os, sys, json, torch, random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from common import SYSTEM
M = "Qwen/Qwen2.5-3B-Instruct"; out = sys.argv[1]
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).cuda()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
rows = [json.loads(l) for l in open(os.path.join(os.environ.get("ISIR_DATA", "data"), "queryner_train.jsonl"))]
def enc(r):
    p = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Query: {r['query']}"}], tokenize=False, add_generation_prompt=True)
    t = json.dumps({"attributes": r["gold"]}) + tok.eos_token
    pi, ti = tok(p, add_special_tokens=False).input_ids, tok(t, add_special_tokens=False).input_ids
    return pi + ti, [-100] * len(pi) + ti
data = [enc(r) for r in rows]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
B, EP = 32, 2; steps = EP * (len(data) // B); model.train(); step = 0
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / steps))
for ep in range(EP):
    random.Random(ep).shuffle(data)
    for i in range(0, len(data) - B + 1, B):
        batch = data[i:i + B]; L = max(len(x) for x, _ in batch)
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch]).cuda()
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch]).cuda()
        att = (torch.arange(L)[None, :] < torch.tensor([len(x) for x, _ in batch])[:, None]).cuda()
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward(); opt.step(); sched.step(); opt.zero_grad(); step += 1
        if step % 50 == 0: print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f}", flush=True)
model.save_pretrained(out); tok.save_pretrained(out); print("DONE lora", out)
