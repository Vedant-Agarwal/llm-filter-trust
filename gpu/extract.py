"""Usage: extract.py MODEL OUT.jsonl [--lora PATH] queries.jsonl ...   -> per query: greedy + 5 samples (T=0.7)."""
import sys, json, argparse
from common import chat, sampling, parse, load_queries
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--lora"); ap.add_argument("files", nargs="+")
a = ap.parse_args()
from vllm import LLM, SamplingParams
kw = dict(model=a.model, gpu_memory_utilization=0.9, max_model_len=2048, enable_prefix_caching=True)
if a.lora: kw.update(enable_lora=True, max_lora_rank=32)
llm = LLM(**kw); tok = llm.get_tokenizer()
qs = load_queries(a.files); prompts = [chat(tok, q["query"]) for q in qs]
lr = None
if a.lora:
    from vllm.lora.request import LoRARequest
    lr = LoRARequest("qner", 1, a.lora)
g = llm.generate(prompts, sampling(SamplingParams, 1, 0.0), lora_request=lr)
s = llm.generate(prompts, sampling(SamplingParams, 5, 0.7), lora_request=lr)
bad = 0
with open(a.out, "w") as f:
    for q, go, so in zip(qs, g, s):
        greedy = parse(go.outputs[0].text); samples = [parse(o.text) for o in so.outputs]
        bad += greedy is None
        f.write(json.dumps({**q, "greedy": greedy, "samples": samples}) + "\n")
print("DONE extract", a.model, a.lora, len(qs), "unparseable greedy:", bad)
