"""The full grid: 3 extractors x 2 retailers x 3 rankers = 18 runs. Reports means, drops, sign-flip p-values, title-fallback and
gating loss avoided, oracle gains, and the LoRA vs 72B drop on shared queries.  Fixed seed."""
import collections, itertools, numpy as np
from lib import *
DS = {d: load_cands(d) for d in RETAILERS}; EXS = {e: load_extract(e) for e in EXTRACTORS}; rng = np.random.default_rng(0)
def safety_table(EX, Q, K=50):
    cov, cons = collections.defaultdict(list), collections.defaultdict(list)
    for q in Q:
        r = EX.get(q["qid"])
        if not r or not r.get("greedy"): continue
        C = q["cands"]; top = [C[j] for j in np.argsort(-hybrid(C))[:K]]
        for a in r["greedy"]:
            t = a["type"]
            if t not in TYPES: continue
            cov[t] += [bool(c["fields"].get(t, "").strip()) for c in top]
            cons[t] += [lex(a["value"], c["fields"].get(t, "")) for c in top if lex(a["value"], c.get("title", ""))]
    return {t: float(np.mean(cov[t])) * (float(np.mean(cons[t])) if len(cons[t]) >= 10 else 0.0) for t in cov}
def run(e, d, base_name):
    EX, Q = EXS[e], DS[d]; SAF = safety_table(EX, Q); per = collections.defaultdict(list); qs = []; n_attr = 0
    for q in Q:
        r = EX.get(q["qid"])
        if not r or not r.get("greedy"): continue
        at = [a for a in r["greedy"] if a["type"] in TYPES]
        if not at: continue
        C = q["cands"]; g = np.array([c["gain"] for c in C])
        if g.max() <= 0: continue
        bm, de = norm01([c["bm25"] for c in C]), norm01([c["dense"] for c in C])
        base = {"hybrid": 0.5 * bm + 0.5 * de, "bm25": bm + 1e-3 * de, "dense": de + 1e-3 * bm}[base_name]; order = np.argsort(-base)
        M = [np.array([lex(a["value"], c["fields"].get(a["type"], "")) for c in C]) for a in at]
        T = [np.array([lex(a["value"], c["fields"].get(a["type"], "")) or lex(a["value"], c.get("title", "")) for c in C]) for a in at]
        def ev(ms):
            keep = np.logical_and.reduce(ms) if ms else np.ones(len(C), bool)
            return ndcg([g[j] for j in order if keep[j]], list(g))
        qs.append(q["qid"]); n_attr += len(at); per["ignore"].append(ev([])); per["hard"].append(ev(M)); per["title"].append(ev(T))
        cur = list(range(len(at)))
        while cur and np.logical_and.reduce([M[i] for i in cur]).sum() < 10:
            cur.sort(key=lambda i: PRIO.index(at[i]["type"])); cur = cur[1:]
        per["relax"].append(ev([M[i] for i in cur]))
        safe = [i for i, a in enumerate(at) if SAF.get(a["type"], 0) >= 0.5]
        per["gate"].append(ev([M[i] for i in safe])); per["gate_title"].append(ev([T[i] for i in safe]))
        per["oracle"].append(max(ev([M[i] for i in s]) for k in range(len(at) + 1) for s in itertools.combinations(range(len(at)), k)) if len(at) <= 4 else ev([]))
    b = np.array(per["ignore"]); n = len(b); o = {"n": n, "n_attrs": n_attr, "ignore": 100 * b.mean()}
    for k in ("hard", "title", "relax", "gate", "gate_title", "oracle"):
        v = np.array(per[k]); df = v - b; fl = np.array([np.mean(df * rng.choice([-1, 1], n)) for _ in range(2000)])
        o[k] = {"mean": 100 * v.mean(), "diff": 100 * df.mean(), "p": float((np.sum(np.abs(fl) >= abs(df.mean())) + 1) / 2001)}
    o["avoid_title"] = 100 * (1 - o["title"]["diff"] / o["hard"]["diff"]); o["avoid_gate_title"] = 100 * (1 - o["gate_title"]["diff"] / o["hard"]["diff"])
    print(e, d, base_name, "n", n, f"ignore {o['ignore']:.2f} hard {o['hard']['mean']:.2f} ({o['hard']['diff']:+.2f}) relax {o['relax']['diff']:+.2f} p={o['relax']['p']:.4f}"
          f" oracle {o['oracle']['diff']:+.2f} avoided: title {o['avoid_title']:.1f}% gate+title {o['avoid_gate_title']:.1f}%", flush=True)
    return o, dict(zip(qs, per["hard"])), dict(zip(qs, per["ignore"]))
out, H = {}, {}
for e in EXTRACTORS:
    for d in RETAILERS:
        for bn in RANKERS:
            o, h, i = run(e, d, bn); out[f"{e}|{d}|{bn}"] = o; H[(e, d, bn)] = (h, i)
shared = {}
for d in RETAILERS:
    hl, il = H[("qwen3b_lora", d, "hybrid")]; h7, i7 = H[("qwen72", d, "hybrid")]; cm = set(hl) & set(h7)
    shared[d] = {"n": len(cm), "lora_drop": 100 * np.mean([il[q] - hl[q] for q in cm]), "qwen72_drop": 100 * np.mean([i7[q] - h7[q] for q in cm])}
    print("shared", d, shared[d])
drops = [-o["hard"]["diff"] for o in out.values()]
summary = {"n_runs": len(out), "drop_min": min(drops), "drop_max": max(drops), "avoid_gate_title_min": min(o["avoid_gate_title"] for o in out.values()),
           "avoid_gate_title_max": max(o["avoid_gate_title"] for o in out.values()), "oracle_max": max(o["oracle"]["diff"] for o in out.values()),
           "oracle_max_hybrid": max(o["oracle"]["diff"] for k, o in out.items() if k.endswith("hybrid"))}
print(summary); save("grid", {"runs": out, "shared": shared, "summary": summary})
