"""All 38 (extractor, retailer, type) cells on raw values: safety vs kept, Spearman overall / where a field exists / within
retailer, and the rule 'safe if safety >= cut' at cut-offs 0.4, 0.5, 0.6, with all labels and with potentially lenient
products removed.  Same cell definition as safety_check.py, which prints two decimals; the rule uses unrounded values."""
import collections, numpy as np
from scipy.stats import spearmanr
from lib import *
DS = {d: load_cands(d) for d in RETAILERS}; cells = []
for e in EXTRACTORS:
    EX = load_extract(e)
    for d, Q in DS.items():
        ag = collections.defaultdict(lambda: {"cov": [], "cons": [], "kept": [], "keptL": []})
        for q in Q:
            r = EX.get(q["qid"])
            if not r or not r.get("greedy"): continue
            C = q["cands"]; top = [C[j] for j in np.argsort(-hybrid(C))[:50]]
            g = [c["gain"] for c in C]; mx = max(g); rel = [c for c in C if c["gain"] == mx] if mx > 0 else []
            len_keys = set()
            for a in r["greedy"]:
                t = a["type"]
                if t in ("color", "material"):
                    L = COLORS if t == "color" else MATS
                    for c in rel:
                        f = c["fields"].get(t, ""); fv = set(toks(f)) & L; qv = set(toks(a["value"])) & L
                        if fv and qv and not (fv & qv) and not lex(a["value"], c.get("title", "")): len_keys.add(c["pid"])
            for a in r["greedy"]:
                t = a["type"]
                if t == "other": continue
                A = ag[t]; A["cov"] += [bool(c["fields"].get(t, "").strip()) for c in top]
                A["cons"] += [lex(a["value"], c["fields"].get(t, "")) for c in top if lex(a["value"], c.get("title", ""))]
                A["kept"] += [lex(a["value"], c["fields"].get(t, "")) for c in rel]
                A["keptL"] += [lex(a["value"], c["fields"].get(t, "")) for c in rel if c["pid"] not in len_keys]
        for t, A in ag.items():
            if len(A["kept"]) < 30: continue
            cov = float(np.mean(A["cov"])); cons = float(np.mean(A["cons"])) if len(A["cons"]) >= 10 else 0.0
            cells.append(dict(extractor=e, retailer=d, type=t, coverage=cov, consistency=cons, safety=cov * cons, kept=float(np.mean(A["kept"])),
                              kept_lenient_out=float(np.mean(A["keptL"])) if A["keptL"] else 0.0, n_rel=len(A["kept"]), n_title=len(A["cons"])))
def sp(rows, nd=None):
    f = (lambda x: x) if nd is None else (lambda x: round(x, nd))
    return float(spearmanr([f(c["safety"]) for c in rows], [f(c["kept"]) for c in rows]).correlation)
nt = [c for c in cells if c["coverage"] > 0]
out = {"n_cells": len(cells), "n_field": len(nt), "n_nofield": len(cells) - len(nt), "spearman_all": sp(cells), "spearman_field": sp(nt), "spearman_field_inputs_rounded_2dp": sp(nt, 2)}
for d in RETAILERS:
    rs = [c for c in nt if c["retailer"] == d]; out[f"spearman_field_{d}"] = sp(rs); out[f"n_field_{d}"] = len(rs)
out["rule"] = {}
for cut in (0.4, 0.5, 0.6):
    m = [c for c in cells if (c["safety"] >= cut) != (c["kept"] >= 0.5)]; mL = [c for c in cells if (c["safety"] >= cut) != (c["kept_lenient_out"] >= 0.5)]
    key = lambda c: [c["extractor"], c["retailer"], c["type"], round(c["safety"], 4), round(c["kept"], 4)]
    out["rule"][str(cut)] = {"correct": len(cells) - len(m), "correct_field": len(nt) - len(m), "misses": [key(c) for c in m], "correct_lenient_out": len(cells) - len(mL)}
    print(f"cut {cut}: {len(cells) - len(m)}/{len(cells)} correct ({len(nt) - len(m)}/{len(nt)} where a field exists), lenient out {len(cells) - len(mL)}/{len(cells)}; misses {[key(c) for c in m]}")
out["cells"] = cells
print({k: v for k, v in out.items() if k.startswith(("spearman", "n_"))})
save("rules", out)
