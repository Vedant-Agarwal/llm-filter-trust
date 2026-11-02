"""Dataset and overlap counts quoted in Section III, and the attribute-count ratios in Section V."""
import collections, json, numpy as np, pandas as pd
from lib import *
out = {}; E = {d: load_cands(d) for d in RETAILERS}
out["esci_queries"] = len(E["esci"]); out["esci_mean_judged"] = float(np.mean([len(q["cands"]) for q in E["esci"]]))
out["esci_min_judged"] = min(len(q["cands"]) for q in E["esci"])
p = pd.read_csv(f"{WANDS}/dataset/product.csv", sep="\t"); q = pd.read_csv(f"{WANDS}/dataset/query.csv", sep="\t"); l = pd.read_csv(f"{WANDS}/dataset/label.csv", sep="\t")
K = set(); KF = set()
for s in p.product_features:
    for kv in str(s).split("|"):
        if ":" in kv:
            k, v = kv.split(":", 1); k = k.strip().lower(); v = v.strip()
            if v: K.add(k)
            if v and v.lower() not in ("nan", "none", "n/a"): KF.add(k)
out.update(wands_products=len(p), wands_queries=len(q), wands_judgments=len(l), wands_judgments_per_query=len(l) / len(q), wands_feature_keys=len(K), wands_feature_keys_used_by_index=len(KF), wands_labels=l.label.value_counts().to_dict())
tr = [json.loads(x) for x in open(f"{DATA}/queryner_train.jsonl")]; te = [json.loads(x) for x in open(f"{DATA}/queryner_test.jsonl")]
out.update(queryner_train=len(tr), queryner_test=len(te))
norm = lambda s: " ".join(s.lower().split())
T = {norm(r["query"]) for r in tr}; L = load_extract("qwen3b_lora"); es = {norm(q["query"]): bool((L.get(q["qid"]) or {}).get("greedy")) for q in E["esci"]}
ov = set(es) & T
out.update(esci_unique_queries=len(es), overlap_queryner_train=len(ov), overlap_share=100 * len(ov) / len(es), overlap_with_attrs=sum(es[x] for x in ov))
# attribute counts over the evaluated queries (at least one attribute of a known type and a positive label)
cnt = collections.defaultdict(dict)
for e in EXTRACTORS:
    EX = load_extract(e)
    out[f"rows_{e}"] = len(EX)
    for d in RETAILERS:
        n = nq = 0
        for x in E[d]:
            r = EX.get(x["qid"]); at = [a for a in (r or {}).get("greedy") or [] if a["type"] in TYPES]
            if at and max(c["gain"] for c in x["cands"]) > 0: n += len(at); nq += 1
        cnt[e][d] = {"attrs": n, "queries": nq}
out["attributes"] = cnt
out["attr_ratio_lora_vs_72b"] = {d: cnt["qwen3b_lora"][d]["attrs"] / cnt["qwen72"][d]["attrs"] for d in RETAILERS}
print(json.dumps({k: v for k, v in out.items()}, indent=0, default=float)); save("stats", out)
