"""Why do hard filters drop relevant products? For every (query, extracted attribute) and every top-relevance product
in the pool, classify: kept (field matches) | dropped: field empty | dropped: field present, different wording |
dropped: attribute not in query (extraction error, judged against QueryNER-style gold where available) .
Also reports the share recovered by embedding soft-matching at threshold tau.
Usage: loss_decomp.py EXTRACT.jsonl [--tau 0.8]"""
import argparse, json, os, pickle, re, collections
DATA = os.environ.get("ISIR_DATA", "data")

import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("extract"); ap.add_argument("--tau", type=float, default=0.80); A = ap.parse_args()
SYN = {"grey": "gray"}
def toks(s): return [SYN.get(t, t) for t in re.findall(r"[a-z0-9]+", str(s).lower())]
def lex(v, f):
    v, f = toks(v), set(toks(f))
    return bool(v) and bool(f) and all(t in f or t.rstrip("s") in f or t + "s" in f for t in v)
EX = {json.loads(l)["qid"]: json.loads(l) for l in open(A.extract)}
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); cache = {}
def E(texts):
    new = [t for t in set(texts) if t not in cache]
    if new:
        for t, e in zip(new, emb.encode(new, batch_size=512, normalize_embeddings=True, show_progress_bar=False)): cache[t] = e
    return np.array([cache[t] for t in texts])
def parts(f): return [p.strip() for p in re.split(r"[;,/|&]| and ", str(f)) if p.strip()][:12]
def soft(v, f, tau):
    ps = parts(f)
    if not ps: return False
    return float((E(ps) @ E([v.lower()])[0]).max()) >= tau
for d in ("esci", "wands"):
    Q = pickle.load(open(f"{DATA}/{d}_cands.pkl", "rb"))
    stats = collections.Counter(); by_type = collections.defaultdict(collections.Counter)
    for q in Q:
        r = EX.get(q["qid"])
        if not r or not r.get("greedy"): continue
        g = [c["gain"] for c in q["cands"]]; top = max(g)
        if top <= 0: continue
        rel = [c for c in q["cands"] if c["gain"] == top]
        for a in r["greedy"]:
            if a["type"] == "other": continue
            for c in rel:
                f = c["fields"].get(a["type"], "")
                if lex(a["value"], f): k = "kept"
                elif not f.strip(): k = "dropped: field empty"
                elif soft(a["value"], f, A.tau): k = "dropped: wording (recoverable by soft match)"
                else: k = "dropped: field disagrees"
                stats[k] += 1; by_type[a["type"]][k] += 1
    n = sum(stats.values())
    print(f"\n== {d}: relevant-product x attribute pairs: {n}")
    for k, v in stats.most_common(): print(f"  {k:48s} {v:7d}  {100*v/n:5.1f}%")
    for t, c in sorted(by_type.items(), key=lambda x: -sum(x[1].values())):
        m = sum(c.values()); print(f"  [{t:9s}] n={m:6d} kept {100*c['kept']/m:5.1f}%  empty {100*c['dropped: field empty']/m:5.1f}%  wording {100*c['dropped: wording (recoverable by soft match)']/m:5.1f}%  disagrees {100*c['dropped: field disagrees']/m:5.1f}%")
