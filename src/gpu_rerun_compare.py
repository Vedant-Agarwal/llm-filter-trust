"""Saved extractions (paper) vs a separate GPU re-run. Usage: gpu_rerun_compare.py RERUN_DIR; with results/gpu_rerun/ from
ISIR_EXTRACTIONS=RERUN_DIR ISIR_RESULTS=results/gpu_rerun python src/{table1,rules,grid}.py. The paper's numbers stay on the saved run."""
import json, re, sys
from lib import *
D = sys.argv[1]; J = lambda p: json.load(open(p)); T0, T1 = J("results/table1.json"), J("results/gpu_rerun/table1.json"); G0, G1 = J("results/grid.json"), J("results/gpu_rerun/grid.json"); R0, R1 = J("results/rules.json"), J("results/gpu_rerun/rules.json")
norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower())); S = lambda g: sorted((a["type"], norm(a["value"])) for a in (g or []))
print("Table I (saved -> re-run)")
for e in EXTRACTORS: print(f"  {e:12s} P {T0[e]['precision']:.3f} -> {T1[e]['precision']:.3f}  R {T0[e]['recall']:.3f} -> {T1[e]['recall']:.3f}  F1 {T0[e]['f1']:.3f} -> {T1[e]['f1']:.3f}")
print("Greedy agreement per query (identical attribute sets after normalization; byte-identical greedy JSON)")
for e in EXTRACTORS:
    a = {json.loads(l)["qid"]: json.loads(l) for l in open(f"extractions/extract_{e}.jsonl")}; b = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/extract_{e}.jsonl")}
    assert a.keys() == b.keys(); same = sum(S(a[q]["greedy"]) == S(b[q]["greedy"]) for q in a); byte = sum(a[q]["greedy"] == b[q]["greedy"] for q in a)
    sub = {p: [q for q in a if q.startswith(p)] for p in ("esci_", "wands_", "qner_")}
    print(f"  {e:12s} {same}/{len(a)} ({100 * same / len(a):.2f}%) sets; {byte}/{len(a)} byte; " + " ".join(f"{p[:-1]} {100 * sum(S(a[q]['greedy']) == S(b[q]['greedy']) for q in v) / len(v):.2f}%" for p, v in sub.items()))
print("Hard-filter drop, hybrid (saved -> re-run)")
for e in EXTRACTORS:
    for d in RETAILERS:
        x, y = G0["runs"][f"{e}|{d}|hybrid"], G1["runs"][f"{e}|{d}|hybrid"]; print(f"  {e:12s} {d:5s} n {x['n']}->{y['n']} drop {-x['hard']['diff']:.2f} -> {-y['hard']['diff']:.2f}  gate+title avoided {x['avoid_gate_title']:.1f}% -> {y['avoid_gate_title']:.1f}%")
print("Largest absolute change in drop over all 18 runs:", round(max(abs(G0["runs"][k]["hard"]["diff"] - G1["runs"][k]["hard"]["diff"]) for k in G0["runs"]), 2))
print("Grid summary:", {k: f"{G0['summary'][k]:.2f}->{G1['summary'][k]:.2f}" for k in G0["summary"]})
print("Rule correct at 0.4/0.5/0.6:", [R0["rule"][c]["correct"] for c in ("0.4", "0.5", "0.6")], "->", [R1["rule"][c]["correct"] for c in ("0.4", "0.5", "0.6")], "cells", R0["n_cells"], "->", R1["n_cells"])
print("Spearman field:", round(R0["spearman_field"], 3), "->", round(R1["spearman_field"], 3))
