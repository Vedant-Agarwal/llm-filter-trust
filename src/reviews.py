"""Camera-ready analyses requested by the ISIR-eCom 2026 reviewers (S63207).
 R-A per-cell counts (queries, relevant product-attribute pairs, title matches) for Table III
 R-B ablation of the safety score: field exists / coverage only / title-field agreement only / product, on all cells and
     on cells where the field exists; rule accuracy against kept>=0.5 and kept>=0.8; false approvals; query-bootstrap
     probability that an approved cell falls below each target
 R-C held-out queries: tune the cut-off on a random half of the queries, test on the other half (200 splits)
 R-D policies side by side on identical queries: ignore, hard, fixed-order relaxation, title fallback, gating by each
     score variant + title fallback; nDCG@10, relevant retention, share of queries with an active filter
 R-E ESCI results after removing queries that occur in QueryNER's training split (LoRA training data)
Usage: reviews.py extractions/extract_A.jsonl [extractions/extract_B.jsonl ...] [--queryner-train FILE] [--base hybrid]
Same matching, candidate pools and safety definition as safety_check.py / depth.py (unchanged)."""
import argparse, json, os, pickle, re, math, collections
DATA = os.environ.get("ISIR_DATA", "data")
import numpy as np
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("extracts", nargs="+"); ap.add_argument("--queryner-train", default=None)
ap.add_argument("--base", default="hybrid"); ap.add_argument("--splits", type=int, default=200); A = ap.parse_args()
SYN = {"grey": "gray"}
TYPES = ["color", "brand", "material", "shape", "style", "size", "pattern"]
PRIO = ["brand", "size", "material", "style", "shape", "pattern", "color"]
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
qn = set()
if A.queryner_train:
    qn = {" ".join(toks(json.loads(l)["query"])) for l in open(A.queryner_train)}
DS = {d: pickle.load(open(f"{DATA}/{d}_cands.pkl", "rb")) for d in ("esci", "wands")}
QTEXT = {}
for d in DS:
    for l in open(f"{DATA}/{d}_queries.jsonl"):
        r = json.loads(l); QTEXT[(d, r["qid"])] = r.get("query", "")
rng = np.random.default_rng(0)

# ---------- per (extractor, retailer, query, attribute) records: label-free stats + label-based kept ----------
REC = []      # dict ex, d, qid, type, cov(list bool), cons(list bool), kept(list bool)
QS = {}       # (ex, d) -> list of per-query policy inputs
for ex_path in A.extracts:
    ex = ex_path.split("/")[-1].replace("extract_", "").replace(".jsonl", "")
    EX = {json.loads(l)["qid"]: json.loads(l) for l in open(ex_path)}
    for d, Q in DS.items():
        QS[(ex, d)] = []
        for q in Q:
            r = EX.get(q["qid"])
            if not r or not r.get("greedy"): continue
            C = q["cands"]; g = np.array([c["gain"] for c in C])
            b = 0.5 * norm01([c["bm25"] for c in C]) + 0.5 * norm01([c["dense"] for c in C])
            top = [C[j] for j in np.argsort(-b)[:50]]
            mx = g.max(); rel = [c for c in C if c["gain"] == mx] if mx > 0 else []
            for a in r["greedy"]:
                t = a["type"]
                if t == "other": continue
                REC.append({"ex": ex, "d": d, "qid": q["qid"], "type": t,
                            "cov": [bool(c["fields"].get(t, "").strip()) for c in top],
                            "cons": [lex(a["value"], c["fields"].get(t, "")) for c in top if lex(a["value"], c.get("title", ""))],
                            "kept": [lex(a["value"], c["fields"].get(t, "")) for c in rel]})
            at = [a for a in r["greedy"] if a["type"] in TYPES]
            if not at or mx <= 0: continue
            bm, de = norm01([c["bm25"] for c in C]), norm01([c["dense"] for c in C])
            base = {"hybrid": 0.5 * bm + 0.5 * de, "bm25": bm + 1e-3 * de, "dense": de + 1e-3 * bm}[A.base]
            QS[(ex, d)].append({"qid": q["qid"], "at": at, "g": g, "order": np.argsort(-base),
                                "M": [np.array([lex(a["value"], c["fields"].get(a["type"], "")) for c in C]) for a in at],
                                "T": [np.array([lex(a["value"], c["fields"].get(a["type"], "")) or lex(a["value"], c.get("title", "")) for c in C]) for a in at]})

def cells(recs):
    """aggregate records into (ex, d, type) cells exactly as safety_check.py (>=30 relevant pairs; consistency 0 if <10 title matches)"""
    agg = collections.defaultdict(lambda: {"cov": [], "cons": [], "kept": [], "q": set()})
    for r in recs:
        X = agg[(r["ex"], r["d"], r["type"])]
        X["cov"] += r["cov"]; X["cons"] += r["cons"]; X["kept"] += r["kept"]; X["q"].add(r["qid"])
    out = {}
    for k, X in agg.items():
        if len(X["kept"]) < 30: continue
        cov = float(np.mean(X["cov"])); cons = float(np.mean(X["cons"])) if len(X["cons"]) >= 10 else 0.0
        out[k] = {"exists": 1.0 if cov > 0 else 0.0, "coverage": cov, "agreement": cons, "safety": cov * cons,
                  "kept": float(np.mean(X["kept"])), "n_q": len(X["q"]), "n_pairs": len(X["kept"]), "n_title": len(X["cons"])}
    return out

ALL = cells(REC)
PRED = ["exists", "coverage", "agreement", "safety"]
print("== R-A per-cell counts (hybrid top-50 pool for the label-free score; relevant = top-gain products)")
print(f"{'extractor':12s} {'ret':5s} {'type':9s} {'queries':>7s} {'relpairs':>8s} {'titlem':>6s} {'cov':>5s} {'agree':>5s} {'SAFE':>5s} {'kept':>5s}")
for k in sorted(ALL, key=lambda k: (k[0], k[1], -ALL[k]["safety"])):
    c = ALL[k]
    print(f"{k[0]:12s} {k[1]:5s} {k[2]:9s} {c['n_q']:7d} {c['n_pairs']:8d} {c['n_title']:6d} {c['coverage']:5.2f} {c['agreement']:5.2f} {c['safety']:5.2f} {c['kept']:5.2f}")
print(f"cells: {len(ALL)}; with a field: {sum(c['exists'] > 0 for c in ALL.values())}")

print("\n== R-B score ablation (cut-off 0.5 for every continuous score; 'exists' = any field present)")
for scope, keys in (("all cells", list(ALL)), ("cells with a field", [k for k in ALL if ALL[k]["exists"] > 0])):
    print(f"-- {scope} (n={len(keys)})")
    kept = np.array([ALL[k]["kept"] for k in keys])
    for p in PRED:
        s = np.array([ALL[k][p] for k in keys])
        rho = spearmanr(s, kept).correlation if len(set(s)) > 1 else float("nan")
        line = f"  {p:9s} Spearman {rho:5.2f}"
        for tgt in (0.5, 0.8):
            appr = s >= 0.5; ok = kept >= tgt
            line += f" | target kept>={tgt}: correct {int(np.sum(appr == ok))}/{len(keys)}, approved {int(appr.sum())}, false approvals {int(np.sum(appr & ~ok))}, missed {int(np.sum(~appr & ok))}"
        print(line)

# query-level bootstrap: for each cell the rule approves, P(kept below target) over resampled queries
print("\n== R-B2 approved cells (safety>=0.5): kept with 95% query-bootstrap CI, P(kept<0.5), P(kept<0.8)")
byq = collections.defaultdict(lambda: collections.defaultdict(list))
for r in REC: byq[(r["ex"], r["d"], r["type"])][r["qid"]] += r["kept"]
for k in sorted(ALL, key=lambda k: (k[0], k[1], k[2])):
    if ALL[k]["safety"] < 0.5: continue
    Ls = list(byq[k].values()); n = len(Ls)
    bs = []
    for _ in range(1000):
        idx = rng.integers(0, n, n); flat = [x for i in idx for x in Ls[i]]
        bs.append(np.mean(flat) if flat else np.nan)
    bs = np.array(bs)
    print(f"  {k[0]:12s} {k[1]:5s} {k[2]:9s} kept {ALL[k]['kept']:.2f} [{np.nanpercentile(bs,2.5):.2f},{np.nanpercentile(bs,97.5):.2f}]  P<0.5 {np.mean(bs<0.5):.2f}  P<0.8 {np.mean(bs<0.8):.2f}")

print(f"\n== R-C held-out queries: {A.splits} random 50/50 query splits (same split for all extractors); cut-off tuned on half A (grid 0.05..0.95, target kept>=0.5), tested on half B")
qids = sorted({(r["d"], r["qid"]) for r in REC})
byid = collections.defaultdict(list)
for r in REC: byid[(r["d"], r["qid"])].append(r)
res = collections.defaultdict(list)
grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
for s in range(A.splits):
    perm = np.random.default_rng(1000 + s).permutation(len(qids)); half = len(qids) // 2
    CA = cells([r for i in perm[:half] for r in byid[qids[i]]]); CB = cells([r for i in perm[half:] for r in byid[qids[i]]])
    common = [k for k in CA if k in CB]
    def acc(Cx, tau, ks): return np.mean([(Cx[k]["safety"] >= tau) == (Cx[k]["kept"] >= 0.5) for k in ks])
    tau = max(grid, key=lambda t: (acc(CA, t, common), -abs(t - 0.5)))
    res["tau"].append(tau); res["n"].append(len(common))
    res["acc_tuned"].append(acc(CB, tau, common)); res["acc_fixed"].append(acc(CB, 0.5, common))
    fe = [k for k in common if CB[k]["exists"] > 0]
    res["acc_tuned_field"].append(acc(CB, tau, fe)); res["acc_fixed_field"].append(acc(CB, 0.5, fe))
    res["fa_tuned"].append(sum(CB[k]["safety"] >= tau and CB[k]["kept"] < 0.5 for k in common))
    res["fa_fixed"].append(sum(CB[k]["safety"] >= 0.5 and CB[k]["kept"] < 0.5 for k in common))
    res["fa80_fixed"].append(sum(CB[k]["safety"] >= 0.5 and CB[k]["kept"] < 0.8 for k in common))
for k, v in res.items():
    v = np.array(v, float); print(f"  {k:16s} mean {v.mean():.3f}  [2.5%,97.5%] [{np.percentile(v,2.5):.3f},{np.percentile(v,97.5):.3f}]")
print("  tuned cut-offs:", dict(collections.Counter(res["tau"]).most_common(6)))

print("\n== R-D policies on identical queries (base=%s): nDCG@10 | relevant kept (share of top-gain products the filters keep) | queries with an active filter" % A.base)
SAFE = {k: v for k, v in ALL.items()}
def policies(P, ex, d):
    at, M, T = P["at"], P["M"], P["T"]
    def sc(t, p): c = SAFE.get((ex, d, t)); return c[p] if c else 0.0
    cur = list(range(len(at)))
    while cur and np.logical_and.reduce([M[i] for i in cur]).sum() < 10:
        cur.sort(key=lambda i: PRIO.index(at[i]["type"])); cur = cur[1:]
    out = {"ignore": [], "hard filter": M, "title fallback (all attrs)": T, "fixed-order relaxation": [M[i] for i in cur]}
    for p in PRED:
        keep = [i for i, a in enumerate(at) if sc(a["type"], p) >= 0.5]
        out[f"gate:{p} + fallback"] = [T[i] for i in keep]
    out["gate:safety, no fallback"] = [M[i] for i, a in enumerate(at) if sc(a["type"], "safety") >= 0.5]
    return out
def run_policies(Qlist, ex, d, label):
    agg = collections.defaultdict(lambda: {"nd": [], "ret": [], "act": []})
    for P in Qlist:
        g = P["g"]; rel = g == g.max()
        for k, masks in policies(P, ex, d).items():
            keep = np.logical_and.reduce(masks) if masks else np.ones(len(g), bool)
            agg[k]["nd"].append(ndcg([g[j] for j in P["order"] if keep[j]], list(g)))
            agg[k]["ret"].append(keep[rel].mean()); agg[k]["act"].append(bool(masks))
    n = len(Qlist); b0 = np.array(agg["ignore"]["nd"])
    print(f"-- {ex} {d} {label} (n={n})")
    for k, v in agg.items():
        nd = np.array(v["nd"]); diff = nd - b0
        bd = [diff[rng.integers(0, n, n)].mean() for _ in range(1000)]
        print(f"  {k:28s} nDCG {100*nd.mean():6.2f}  vs ignore {100*diff.mean():+6.2f} [{100*np.percentile(bd,2.5):+6.2f},{100*np.percentile(bd,97.5):+6.2f}]  rel kept {100*np.mean(v['ret']):5.1f}%  active {100*np.mean(v['act']):5.1f}%")
for (ex, d), Ql in QS.items(): run_policies(Ql, ex, d, "all queries")

if qn:
    print("\n== R-E ESCI queries that occur in QueryNER train (normalized text match)")
    for (ex, d), Ql in QS.items():
        if d != "esci": continue
        ov = [P for P in Ql if " ".join(toks(QTEXT.get(("esci", P["qid"]), ""))) in qn]
        allq = {" ".join(toks(t)) for (dd, _), t in QTEXT.items() if dd == "esci"}
        print(f"  {ex}: {len(ov)}/{len(Ql)} evaluated queries overlap ({sum(x in qn for x in allq)}/{len(allq)} of all ESCI sample queries)")
        run_policies([P for P in Ql if P not in ov], ex, d, "EXCLUDING QueryNER-train overlap")
    # cell table and rule without overlap
    keep_ids = {P["qid"] for (ex, d), Ql in QS.items() if d == "esci" for P in Ql if " ".join(toks(QTEXT.get(("esci", P["qid"]), ""))) not in qn}
    C2 = cells([r for r in REC if r["d"] != "esci" or " ".join(toks(QTEXT.get(("esci", r["qid"]), ""))) not in qn])
    print("  rule (safety>=0.5 vs kept>=0.5) without overlap:", sum((c["safety"] >= 0.5) == (c["kept"] >= 0.5) for c in C2.values()), "/", len(C2))
    for k in sorted(C2):
        if k[1] == "esci": print(f"   {k[0]:12s} esci {k[2]:9s} safety {C2[k]['safety']:.2f} kept {C2[k]['kept']:.2f} (all queries: {ALL[k]['safety']:.2f}/{ALL[k]['kept']:.2f})")
