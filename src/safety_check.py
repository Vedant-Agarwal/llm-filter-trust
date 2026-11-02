"""Label-free filter-safety check. For each (retailer, attribute type) compute, from the catalog and the queries alone
(no relevance labels): coverage = share of candidate products (top-50 by the base ranker) with a non-empty field;
consistency = among products whose TITLE contains the extracted value, share whose FIELD also matches it
(title-field agreement, a label-free proxy for catalog correctness).  safety = coverage x consistency.
Validate against the label-based truth: share of relevant (top-gain) products a hard filter keeps.
Usage: safety_check.py EXTRACT.jsonl [EXTRACT2.jsonl ...]"""
import os, sys, json, pickle, re, collections
DATA = os.environ.get("ISIR_DATA", "data")

import numpy as np
from scipy.stats import spearmanr, pearsonr
SYN = {"grey": "gray"}
def toks(s): return [SYN.get(t, t) for t in re.findall(r"[a-z0-9]+", str(s).lower())]
def lex(v, f):
    v, f = toks(v), set(toks(f))
    return bool(v) and bool(f) and all(t in f or t.rstrip("s") in f or t + "s" in f for t in v)
def norm01(x):
    x = np.asarray(x, float); lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo + 1e-9)
rows = []
for ex_path in sys.argv[1:]:
    EX = {json.loads(l)["qid"]: json.loads(l) for l in open(ex_path)}
    for d in ("esci", "wands"):
        Q = pickle.load(open(f"{DATA}/{d}_cands.pkl", "rb"))
        agg = collections.defaultdict(lambda: {"cov": [], "cons": [], "kept": []})
        for q in Q:
            r = EX.get(q["qid"])
            if not r or not r.get("greedy"): continue
            C = q["cands"]
            base = 0.5 * norm01([c["bm25"] for c in C]) + 0.5 * norm01([c["dense"] for c in C])
            top = [C[j] for j in np.argsort(-base)[:50]]
            g = [c["gain"] for c in C]; mx = max(g)
            rel = [c for c in C if c["gain"] == mx] if mx > 0 else []
            for a in r["greedy"]:
                t = a["type"]
                if t == "other": continue
                A = agg[t]
                A["cov"] += [bool(c["fields"].get(t, "").strip()) for c in top]
                for c in top:                                  # label-free consistency: title says it -> does the field?
                    if lex(a["value"], c.get("title", "")):
                        A["cons"].append(lex(a["value"], c["fields"].get(t, "")))
                A["kept"] += [lex(a["value"], c["fields"].get(t, "")) for c in rel]   # label-based truth
        for t, A in agg.items():
            if len(A["kept"]) < 30: continue
            cov = float(np.mean(A["cov"])); cons = float(np.mean(A["cons"])) if len(A["cons"]) >= 10 else 0.0
            rows.append({"extractor": ex_path.split("/")[-1].replace("extract_", "").replace(".jsonl", ""), "retailer": d, "type": t,
                         "coverage": round(cov, 3), "consistency": round(cons, 3), "safety": round(cov * cons, 3),
                         "kept_relevant": round(float(np.mean(A["kept"])), 3), "n_rel": len(A["kept"]), "n_title": len(A["cons"])})
print(f"{'extractor':12s} {'retailer':7s} {'type':9s} {'cover':>6s} {'consist':>7s} {'SAFETY':>7s} | {'kept(truth)':>11s} n")
for r in sorted(rows, key=lambda r: (r["extractor"], r["retailer"], -r["safety"])):
    print(f"{r['extractor']:12s} {r['retailer']:7s} {r['type']:9s} {r['coverage']:6.2f} {r['consistency']:7.2f} {r['safety']:7.2f} | {r['kept_relevant']:11.2f} {r['n_rel']}")
s = [r["safety"] for r in rows]; k = [r["kept_relevant"] for r in rows]
print(f"\nall cells: n={len(rows)}  Spearman {spearmanr(s, k).correlation:.3f}  Pearson {pearsonr(s, k)[0]:.3f}")
for d in ("esci", "wands"):
    ss = [r["safety"] for r in rows if r["retailer"] == d]; kk = [r["kept_relevant"] for r in rows if r["retailer"] == d]
    if len(ss) > 3: print(f"{d}: n={len(ss)} Spearman {spearmanr(ss, kk).correlation:.3f}")
# transfer of a threshold: pick tau on one retailer (max accuracy of 'safe if safety>=tau' vs truth kept>=0.5), apply to other
def acc(rs, tau): return np.mean([(r["safety"] >= tau) == (r["kept_relevant"] >= 0.5) for r in rs])
for src, tgt in (("esci", "wands"), ("wands", "esci")):
    S = [r for r in rows if r["retailer"] == src]; T = [r for r in rows if r["retailer"] == tgt]
    if not S or not T: continue
    tau = max(np.linspace(0, 1, 41), key=lambda t: acc(S, t))
    print(f"threshold tuned on {src}: tau={tau:.2f} -> {tgt} accuracy {acc(T, tau):.2f} (source {acc(S, tau):.2f})")
