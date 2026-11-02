"""Robustness for the reviewer checklist.
(1) cosine threshold for the 'different wording' class: 0.7 / 0.8 / 0.9 -> loss decomposition shares;
(2) safety cut-off 0.4 / 0.5 / 0.6 -> rule accuracy vs truth and loss avoided by gating + title fallback;
(3) lenient labels: a top-relevance product is 'potentially lenient' for a color/material attribute when its catalog
    field names a DIFFERENT value of that attribute and its title does not contain the queried value (e.g. a black
    dresser judged Exact for 'white dresser'). Count them and recompute the headline hard-filter drop with them
    treated as non-relevant.
Usage: robust.py EXTRACT.jsonl"""
import os, sys, json, pickle, re, math, collections
DATA = os.environ.get("ISIR_DATA", "data")

import numpy as np
from sentence_transformers import SentenceTransformer
SYN = {"grey": "gray"}
TYPES = ["color", "brand", "material", "shape", "style", "size", "pattern"]
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
EX = {json.loads(l)["qid"]: json.loads(l) for l in open(sys.argv[1])}
emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); cache = {}
def E(texts):
    new = [t for t in set(texts) if t not in cache]
    if new:
        for t, e in zip(new, emb.encode(new, batch_size=512, normalize_embeddings=True, show_progress_bar=False)): cache[t] = e
    return np.array([cache[t] for t in texts])
def parts(f): return [p.strip() for p in re.split(r"[;,/|&]| and ", str(f)) if p.strip()][:12]
def maxcos(v, f):
    ps = parts(f)
    return float((E(ps) @ E([v.lower()])[0]).max()) if ps else -1.0

for d in ("esci", "wands"):
    Q = pickle.load(open(f"{DATA}/{d}_cands.pkl", "rb"))
    # (1) wording threshold
    sims, kept_or_empty = [], collections.Counter()
    # (3) lenient
    lenient = total_rel = 0; lenient_keys = set()
    for q in Q:
        r = EX.get(q["qid"])
        if not r or not r.get("greedy"): continue
        g = [c["gain"] for c in q["cands"]]; top = max(g)
        if top <= 0: continue
        rel = [c for c in q["cands"] if c["gain"] == top]
        for a in r["greedy"]:
            t = a["type"]
            if t not in TYPES: continue
            for c in rel:
                f = c["fields"].get(t, "")
                if lex(a["value"], f): kept_or_empty["kept"] += 1
                elif not f.strip(): kept_or_empty["empty"] += 1
                else: sims.append(maxcos(a["value"], f))
                if t in ("color", "material"):
                    total_rel += 1
                    lexset = COLORS if t == "color" else MATS
                    fv = set(toks(f)) & lexset; qv = set(toks(a["value"])) & lexset
                    if fv and qv and not (fv & qv) and not lex(a["value"], c.get("title", "")):
                        lenient += 1; lenient_keys.add((q["qid"], c["pid"]))
    n = sum(kept_or_empty.values()) + len(sims); sims = np.array(sims)
    print(f"\n== {d}: wording-vs-disagree split by cosine threshold (share of all relevant x attribute pairs, n={n})")
    for tau in (0.7, 0.8, 0.9):
        w = (sims >= tau).sum(); print(f"  tau={tau}: wording {100*w/n:5.1f}%  disagrees {100*(len(sims)-w)/n:5.1f}%  (kept {100*kept_or_empty['kept']/n:.1f}%, empty {100*kept_or_empty['empty']/n:.1f}% unchanged)")
    print(f"  potentially lenient labels (color/material): {lenient} of {total_rel} relevant pairs ({100*lenient/max(1,total_rel):.1f}%), {len(lenient_keys)} distinct (query, product)")
    # headline with lenient products treated as non-relevant
    for drop in (False, True):
        ig, hd = [], []
        for q in Q:
            r = EX.get(q["qid"])
            if not r or not r.get("greedy"): continue
            at = [a for a in r["greedy"] if a["type"] in TYPES]
            if not at: continue
            C = q["cands"]; g = np.array([0.0 if drop and (q["qid"], c["pid"]) in lenient_keys else c["gain"] for c in C])
            if g.max() <= 0: continue
            base = 0.5 * norm01([c["bm25"] for c in C]) + 0.5 * norm01([c["dense"] for c in C])
            keep = np.logical_and.reduce([np.array([lex(a["value"], c["fields"].get(a["type"], "")) for c in C]) for a in at])
            order = np.argsort(-base)
            ig.append(ndcg([g[j] for j in order], list(g))); hd.append(ndcg([g[j] for j in order if keep[j]], list(g)))
        print(f"  {'lenient removed' if drop else 'all labels    '}: ignore {100*np.mean(ig):.1f}  hard filter {100*np.mean(hd):.1f}  drop {100*(np.mean(ig)-np.mean(hd)):.1f}  (n={len(ig)})")
