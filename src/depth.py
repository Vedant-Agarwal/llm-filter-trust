"""Depth analyses for the paper: (1) bootstrap 95% CIs + paired significance for ignore vs hard vs relaxation vs
title-fallback repair vs oracle; (2) catalog repair = an attribute 'matches' if the catalog FIELD matches OR, when the
field is empty/disagrees, the product TITLE contains the value; (3) safety-score sensitivity to K and the cut-off.
Usage: depth.py EXTRACT.jsonl [--base hybrid]"""
import argparse, json, os, pickle, re, math, itertools, collections
DATA = os.environ.get("ISIR_DATA", "data")

import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("extract"); ap.add_argument("--base", default="hybrid"); ap.add_argument("--cut", type=float, default=0.5); A = ap.parse_args()
SYN = {"grey": "gray"}
TYPES = ["color", "brand", "material", "shape", "style", "size", "pattern"]
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
EX = {json.loads(l)["qid"]: json.loads(l) for l in open(A.extract)}
rng = np.random.default_rng(0)
PRIO = ["brand", "size", "material", "style", "shape", "pattern", "color"]

def safety_table(d, Q, K=50):
    """label-free: coverage x title-field consistency per attribute type, from catalog + queries only"""
    cov, cons = collections.defaultdict(list), collections.defaultdict(list)
    for q in Q:
        r = EX.get(q["qid"])
        if not r or not r.get("greedy"): continue
        C = q["cands"]; b = 0.5 * norm01([c["bm25"] for c in C]) + 0.5 * norm01([c["dense"] for c in C])
        top = [C[j] for j in np.argsort(-b)[:K]]
        for a in r["greedy"]:
            t = a["type"]
            if t not in TYPES: continue
            cov[t] += [bool(c["fields"].get(t, "").strip()) for c in top]
            cons[t] += [lex(a["value"], c["fields"].get(t, "")) for c in top if lex(a["value"], c.get("title", ""))]
    return {t: float(np.mean(cov[t])) * (float(np.mean(cons[t])) if len(cons[t]) >= 10 else 0.0) for t in cov}

def run(d):
    Q = pickle.load(open(f"{DATA}/{d}_cands.pkl", "rb")); per = collections.defaultdict(list)
    SAF = safety_table(d, Q); print(f"\n[{d}] label-free safety per type:", {t: round(v, 2) for t, v in sorted(SAF.items(), key=lambda x: -x[1])})
    for q in Q:
        r = EX.get(q["qid"])
        if not r or not r.get("greedy"): continue
        at = [a for a in r["greedy"] if a["type"] in TYPES]
        if not at: continue
        C = q["cands"]; g = np.array([c["gain"] for c in C])
        if g.max() <= 0: continue
        bm, de = norm01([c["bm25"] for c in C]), norm01([c["dense"] for c in C])
        base = {"hybrid": 0.5 * bm + 0.5 * de, "bm25": bm + 1e-3 * de, "dense": de + 1e-3 * bm}[A.base]
        M = [np.array([lex(a["value"], c["fields"].get(a["type"], "")) for c in C]) for a in at]
        T = [np.array([lex(a["value"], c["fields"].get(a["type"], "")) or lex(a["value"], c.get("title", "")) for c in C]) for a in at]
        def ev(masks):
            keep = np.logical_and.reduce(masks) if masks else np.ones(len(C), bool)
            return ndcg([g[j] for j in np.argsort(-base) if keep[j]], list(g))
        per["ignore"].append(ev([]))
        per["hard filter"].append(ev(M))
        per["hard filter + title fallback (repair)"].append(ev(T))
        cur = list(range(len(at)))
        while cur and np.logical_and.reduce([M[i] for i in cur]).sum() < 10:
            cur.sort(key=lambda i: PRIO.index(at[i]["type"])); cur = cur[1:]
        per["fixed-order relaxation"].append(ev([M[i] for i in cur]))
        safe = [i for i, a in enumerate(at) if SAF.get(a["type"], 0) >= A.cut]
        per["safety-gated hard filter (safety>=0.5)"].append(ev([M[i] for i in safe]))
        per["safety-gated + title fallback"].append(ev([T[i] for i in safe]))
        per["oracle subset of hard filters"].append(max(ev([M[i] for i in s]) for k in range(len(at) + 1) for s in itertools.combinations(range(len(at)), k)) if len(at) <= 4 else ev([]))
    n = len(per["ignore"]); base = np.array(per["ignore"])
    print(f"\n== {d} (base={A.base}, n={n} queries); nDCG@10 mean [95% bootstrap CI]; paired diff vs ignore [CI], p (paired sign-flip)")
    for k, v in per.items():
        v = np.array(v); bs = [v[rng.integers(0, n, n)].mean() for _ in range(2000)]
        diff = v - base; bd = [diff[rng.integers(0, n, n)].mean() for _ in range(2000)]
        flips = [np.mean(diff * rng.choice([-1, 1], n)) for _ in range(2000)]
        p = (np.sum(np.abs(flips) >= abs(diff.mean())) + 1) / 2001
        print(f"  {k:40s} {100*v.mean():6.2f} [{100*np.percentile(bs,2.5):6.2f},{100*np.percentile(bs,97.5):6.2f}]   diff {100*diff.mean():+7.2f} [{100*np.percentile(bd,2.5):+7.2f},{100*np.percentile(bd,97.5):+7.2f}]  p={p:.4f}")

for d in ("esci", "wands"):
    run(d)
