"""Helpers for the scripts added for this repository (rules, grid, stats, table1, table2, gpu_rerun_compare).
The matching and metric functions are the same as in depth.py, robust.py and safety_check.py."""
import json, math, os, pickle, re
import numpy as np
DATA = os.environ.get("ISIR_DATA", "data"); WANDS = os.environ.get("ISIR_WANDS", "WANDS"); RESULTS = os.environ.get("ISIR_RESULTS", "results"); EXTRACT = os.environ.get("ISIR_EXTRACTIONS", "extractions")
EXTRACTORS = ("qwen3b_lora", "qwen72", "qwen3b_base"); RETAILERS = ("esci", "wands"); RANKERS = ("hybrid", "bm25", "dense")
SYN = {"grey": "gray"}
TYPES = ["color", "brand", "material", "shape", "style", "size", "pattern"]
PRIO = ["brand", "size", "material", "style", "shape", "pattern", "color"]
COLORS = {"black", "white", "gray", "brown", "blue", "green", "red", "beige", "gold", "silver", "navy", "pink", "yellow", "orange",
          "purple", "ivory", "cream", "tan", "teal", "turquoise", "espresso", "walnut", "natural", "clear", "multicolor", "bronze", "copper"}
MATS = {"wood", "metal", "leather", "fabric", "glass", "plastic", "velvet", "cotton", "linen", "marble", "steel", "iron", "rattan",
        "wicker", "polyester", "ceramic", "porcelain", "silicone", "aluminum", "oak", "pine", "bamboo", "wool", "acrylic", "stone"}
def toks(s): return [SYN.get(t, t) for t in re.findall(r"[a-z0-9]+", str(s).lower())]
def lex(v, f):
    v, f = toks(v), set(toks(f))
    return bool(v) and bool(f) and all(t in f or t.rstrip("s") in f or t + "s" in f for t in v)
def norm01(x):
    x = np.asarray(x, float); lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo + 1e-9)
def ndcg(ranked, allg, k=10):
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ranked[:k]))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(allg, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0
def load_extract(e): return {json.loads(l)["qid"]: json.loads(l) for l in open(f"{EXTRACT}/extract_{e}.jsonl")}
def load_cands(d): return pickle.load(open(f"{DATA}/{d}_cands.pkl", "rb"))
def hybrid(C): return 0.5 * norm01([c["bm25"] for c in C]) + 0.5 * norm01([c["dense"] for c in C])
def save(name, obj):
    json.dump(obj, open(f"{RESULTS}/{name}.json", "w"), indent=1, default=float); print("wrote", f"{RESULTS}/{name}.json")
