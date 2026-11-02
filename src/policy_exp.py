"""Filter / boost / ignore policy for LLM-extracted query attributes; train on one retailer, test on another.
Usage: policy_exp.py EXTRACT.jsonl [--types-agnostic]
Needs $ISIR_DATA/esci_cands.pkl and wands_cands.pkl (build_index.py)."""
import argparse, json, os, pickle, re, math, itertools, collections
DATA = os.environ.get("ISIR_DATA", "data")

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

TYPES = ["color", "brand", "material", "shape", "style", "size", "pattern"]
SYN = {"grey": "gray", "colour": "color"}
ap = argparse.ArgumentParser(); ap.add_argument("extract"); ap.add_argument("--types-agnostic", action="store_true")
ap.add_argument("--beta", type=float, default=None); ap.add_argument("--model", default="ridge")
ap.add_argument("--base", default="hybrid", choices=["hybrid", "bm25", "dense"])
ap.add_argument("--boost", default="rank", choices=["score", "rank"])
A = ap.parse_args()

def toks(s):
    return [SYN.get(t, t) for t in re.findall(r"[a-z0-9]+", str(s).lower())]

def matches(value, field):
    v, f = toks(value), set(toks(field))
    if not v or not f: return False
    return all(t in f or t.rstrip("s") in f or (t + "s") in f for t in v)

def norm01(x):
    x = np.asarray(x, float); lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo + 1e-9)

def ndcg(gains_ranked, ideal_gains, k=10):
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains_ranked[:k]))
    ideal = sorted(ideal_gains, reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None

# ---------- load ----------
EX = {}
for l in open(A.extract):
    r = json.loads(l); EX[r["qid"]] = r
DS = {d: pickle.load(open(f"{DATA}/{d}_cands.pkl", "rb")) for d in ("esci", "wands")}

def attrs_for(qid):
    r = EX.get(qid)
    if not r or not r.get("greedy"): return []
    out = []
    for a in r["greedy"]:
        if a["type"] not in TYPES: continue
        nv = " ".join(toks(a["value"]))
        conf = np.mean([any(b["type"] == a["type"] and " ".join(toks(b["value"])) == nv for b in (s or [])) for s in r["samples"]])
        out.append({"type": a["type"], "value": a["value"], "conf": float(conf)})
    return out

# ---------- per query precompute ----------
def prep(q):
    C = q["cands"]
    bm, de = norm01([c["bm25"] for c in C]), norm01([c["dense"] for c in C])
    base = {"hybrid": 0.5 * bm + 0.5 * de, "bm25": bm + 1e-3 * de, "dense": de + 1e-3 * bm}[A.base]
    if A.boost == "rank":   # scale-free: base becomes rank-percentile in [0,1], so a boost means "move up by beta of the list"
        base = 1 - np.argsort(np.argsort(-base)) / max(1, len(base) - 1)
    order = np.argsort(-base)
    gains = np.array([c["gain"] for c in C])
    return {"C": C, "base": base, "order": order, "gains": gains}

def feats(P, a):
    C, order = P["C"], P["order"]
    top = order[:50]
    filled = np.array([bool(C[j]["fields"].get(a["type"], "")) for j in top])
    mt_all = np.array([matches(a["value"], c["fields"].get(a["type"], "")) for c in C])
    mt = mt_all[top]
    fill = filled.mean(); match = mt.mean(); mgf = mt[filled].mean() if filled.any() else 0.0
    surv = math.log1p(mt_all.sum()) / math.log1p(len(C))
    agree10 = mt_all[order[:10]].mean()
    f = [a["conf"], fill, match, mgf, surv, agree10, len(toks(a["value"]))]
    if not A.types_agnostic:
        f += [1.0 if a["type"] == t else 0.0 for t in TYPES]
    return np.array(f), mt_all

def score(P, acts, beta):
    """acts: list of (action, match_mask). returns nDCG@10, exact-recall@100"""
    s = P["base"].copy(); keep = np.ones(len(s), bool)
    for act, m in acts:
        if act == "hard": keep &= m
        elif act == "boost": s = s + beta * m
    idx = [j for j in np.argsort(-s) if keep[j]]
    g = P["gains"]
    nd = ndcg([g[j] for j in idx], list(g))
    top = max(g) if len(g) else 0
    ex = [j for j in range(len(g)) if g[j] == top and top > 0]
    rec = (len(set(idx[:100]) & set(ex)) / len(ex)) if ex else None
    return nd, rec

# ---------- build rows ----------
def build(d):
    Q = []
    for q in DS[d]:
        at = attrs_for(q["qid"])
        if not at: continue
        P = prep(q)
        if not (P["gains"] > 0).any(): continue
        items = []
        for a in at:
            x, m = feats(P, a); items.append({"a": a, "x": x, "m": m})
        Q.append({"qid": q["qid"], "P": P, "items": items})
    return Q

def tune_beta(Q):
    best = None
    for b in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        v = np.nanmean([score(q["P"], [("boost", it["m"]) for it in q["items"]], b)[0] or 0 for q in Q])
        if best is None or v > best[1]: best = (b, v)
    return best[0]

def labels(Q, beta):
    X, Y = [], []
    for q in Q:
        for it in q["items"]:
            base = score(q["P"], [], beta)[0]
            h = score(q["P"], [("hard", it["m"])], beta)[0]; b = score(q["P"], [("boost", it["m"])], beta)[0]
            X.append(it["x"]); Y.append([h - base, b - base])
    return np.array(X), np.array(Y)

def fit(X, Y):
    mk = (lambda: Ridge(alpha=1.0)) if A.model == "ridge" else (lambda: GradientBoostingRegressor(max_depth=2, n_estimators=150, learning_rate=0.05, subsample=0.8, random_state=0))
    mh, mb = mk().fit(X, Y[:, 0]), mk().fit(X, Y[:, 1])
    return lambda x: ("hard" if (ph := mh.predict([x])[0]) >= max(0, (pb := mb.predict([x])[0])) and ph > 0 else ("boost" if pb > 0 else "ignore"))

PRIO = ["brand", "size", "material", "style", "shape", "pattern", "color"]   # fixed relaxation order (drop first)
def evaluate(Q, policy, beta, name):
    res = collections.defaultdict(list)
    for q in Q:
        its = q["items"]
        strat = {
            "ignore (base ranker)": [],
            "always hard-filter": [("hard", it["m"]) for it in its],
            "always boost": [("boost", it["m"]) for it in its],
        }
        # fixed-order relaxation: hard filters, drop lowest-priority attribute until >=10 survivors
        cur = list(its)
        while cur and np.logical_and.reduce([it["m"] for it in cur]).sum() < 10:
            cur.sort(key=lambda it: PRIO.index(it["a"]["type"])); cur = cur[1:]
        strat["fixed-order relaxation"] = [("hard", it["m"]) for it in cur]
        strat["heuristic (conf>=.8 & fill>=.9 -> hard, else boost)"] = [("hard" if it["x"][0] >= .8 and it["x"][1] >= .9 else "boost", it["m"]) for it in its]
        strat[f"learned policy ({name})"] = [(policy(it["x"]), it["m"]) for it in its]
        best = max((score(q["P"], [(a, it["m"]) for a, it in zip(combo, its)], beta)[0] or 0, combo) for combo in itertools.product(["hard", "boost", "ignore"], repeat=min(len(its), 3)) if len(its) <= 3) if len(its) <= 3 else None
        if best: strat["oracle (best per query)"] = [(a, it["m"]) for a, it in zip(best[1], its)]
        for k, acts in strat.items():
            nd, rec = score(q["P"], acts, beta); res[k].append((nd or 0.0, rec))
    out = {}
    for k, v in res.items():
        nd = np.mean([a for a, _ in v]); rc = [b for _, b in v if b is not None]
        out[k] = (round(100 * nd, 2), round(100 * np.mean(rc), 2) if rc else None, len(v))
    return out

if __name__ == "__main__":
    Q = {d: build(d) for d in DS}
    for d in Q: print(d, "queries with >=1 extracted attribute:", len(Q[d]), "| attrs:", sum(len(q["items"]) for q in Q[d]),
                      "| types:", dict(collections.Counter(it["a"]["type"] for q in Q[d] for it in q["items"])))
    for src, tgt in [("esci", "wands"), ("wands", "esci"), ("esci", "esci"), ("wands", "wands")]:
        beta = A.beta or tune_beta(Q[src])
        if src == tgt:   # in-domain: 2-fold by query
            rng = np.random.default_rng(0); idx = rng.permutation(len(Q[src])); half = len(idx) // 2
            tr, te = [Q[src][i] for i in idx[:half]], [Q[src][i] for i in idx[half:]]
        else:
            tr, te = Q[src], Q[tgt]
        X, Y = labels(tr, beta); pol = fit(X, Y)
        r = evaluate(te, pol, beta, f"trained on {src}")
        print(f"\n== train {src} -> test {tgt} (beta={beta}, model={A.model}, types_agnostic={A.types_agnostic}); nDCG@10, Exact-recall@100, n")
        for k, v in r.items(): print(f"  {k:60s} {v}")
