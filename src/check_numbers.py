"""Compare every number claimed in the paper (camera-ready revision v11) with the result files. Run after run_all.sh (needs no data, only results/ and code).
Each row: paper line (tex/paper.tex, revision v11), claim, expected, actual, tolerance, PASS/FAIL/MISSING.
Default tolerance is half a unit in the paper's last printed digit, so PASS means the paper's rounding is right.
Bootstrap CIs and sign-flip p-values use the fixed seed in the code; they get the wider STOCH tolerance.
Provenance: C = recomputed on CPU in this run; G = read from a saved GPU output or log (extraction and LoRA training were not rerun)."""
import json, os, re, sys
R = "results"; ROWS = []; STOCH = 0.3
def J(n): return json.load(open(f"{R}/{n}.json"))
def add(line, what, exp, act, tol=None, prov="C"):
    """exp: number, or string of the paper's printed number (sets the tolerance from its decimals)."""
    if act is None: ROWS.append((line, what, exp, None, tol, prov, "MISSING")); return
    if tol is None:
        s = str(exp); dec = len(s.split(".")[1]) if "." in s else 0; tol = 0.5 * 10 ** -dec
    ok = abs(float(exp) - float(act)) <= tol + 1e-9
    ROWS.append((line, what, exp, round(float(act), 4), tol, prov, "PASS" if ok else "FAIL"))
def claim(line, what, ok, act, prov="C"):
    ROWS.append((line, what, "true", act, 0, prov, "PASS" if ok else "FAIL"))
def depth(path):
    out, d = {}, None
    for l in open(path):
        m = re.match(r"== (\w+) \(base=\w+, n=(\d+)", l)
        if m: d = m.group(1); out[d] = {"n": int(m.group(2))}; continue
        m = re.match(r"\s+(.+?)\s+(-?[\d.]+) \[\s*(-?[\d.]+),\s*(-?[\d.]+)\]\s+diff\s+([-+][\d.]+) \[\s*([-+][\d.]+),\s*([-+][\d.]+)\]\s+p=([\d.]+)", l)
        if m and d: out[d][m.group(1)] = dict(zip(("mean", "lo", "hi", "diff", "dlo", "dhi", "p"), map(float, m.groups()[1:])))
    return out
def loss(path):
    out, d = {}, None
    for l in open(path):
        m = re.match(r"== (\w+):", l)
        if m: d = m.group(1); out[d] = {}; continue
        m = re.match(r"\s+\[(\w+)\s*\] n=\s*\d+ kept\s+([\d.]+)%\s+empty\s+([\d.]+)%\s+wording\s+([\d.]+)%\s+disagrees\s+([\d.]+)%", l)
        if m: out[d][m.group(1)] = dict(zip(("kept", "empty", "wording", "disagree"), map(float, m.groups()[1:])))
    return out
def robust(path):
    out, d = {}, None
    for l in open(path):
        m = re.match(r"== (\w+):", l)
        if m: d = m.group(1); out[d] = {}; continue
        m = re.search(r"tau=([\d.]+): wording\s+([\d.]+)%", l)
        if m: out[d]["w" + m.group(1)] = float(m.group(2))
        m = re.search(r"\(([\d.]+)%\), \d+ distinct", l)
        if m: out[d]["lenient"] = float(m.group(1))
        m = re.search(r"(all labels|lenient removed)\s*: ignore ([\d.]+)\s+hard filter ([\d.]+)\s+drop ([\d.]+)", l)
        if m: out[d]["drop_all" if m.group(1) == "all labels" else "drop_len"] = float(m.group(4))
    return out
def policy(path):
    out = {}
    for block in open(path).read().split("\n== train ")[1:]:
        head = block.split("\n")[0]; m = re.match(r"(\w+) -> test (\w+)", head)
        if m.group(1) == m.group(2): continue
        ig = re.search(r"ignore \(base ranker\)\s+\(np\.float64\(([\d.]+)\)", block); le = re.search(r"learned policy[^(]*\(trained on \w+\)\s+\(np\.float64\(([\d.]+)\)", block)
        out[m.group(2)] = float(le.group(1)) - float(ig.group(1))
    return out
def avoid(d, k="safety-gated + title fallback"): return 100 * (1 - d[k]["diff"] / d["hard filter"]["diff"])
def code(path, pat): return re.search(pat, open(path).read()) is not None
def reviews(path):
    """results/reviews.txt (src/reviews.py): R-A cell counts, R-B ablation, R-C held-out splits, R-D policies, R-E overlap."""
    t = open(path).read(); out = {"cells": {}, "abl": {}, "heldout": {}, "pol": {}, "polx": {}, "overlap": None}
    for m in re.finditer(r"^(\w+)\s+(esci|wands)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", t, re.M):
        out["cells"][m.group(1, 2, 3)] = dict(zip(("q", "pairs", "titlem"), map(int, m.group(4, 5, 6))))
    blk = t.split("-- cells with a field (n=22)")[1].split("==")[0]
    for m in re.finditer(r"^\s+(\w+)\s+Spearman\s+\S+ \| target kept>=0.5: correct (\d+)/\d+, approved (\d+), false approvals (\d+)", blk, re.M):
        out["abl"][m.group(1)] = dict(zip(("correct", "approved", "fa"), map(int, m.group(2, 3, 4))))
    for m in re.finditer(r"^\s+(\w+)\s+mean ([\d.]+)\s+\[2.5%,97.5%\] \[([\d.]+),([\d.]+)\]", t.split("== R-C")[1].split("== R-D")[0], re.M):
        out["heldout"][m.group(1)] = tuple(map(float, m.group(2, 3, 4)))
    for key, part in (("pol", t.split("== R-D")[1].split("== R-E")[0]), ("polx", t.split("== R-E")[1])):
        for b in re.split(r"^-- ", part, flags=re.M)[1:]:
            h = re.match(r"(\w+) (esci|wands)", b); rows = {}
            for m in re.finditer(r"^\s+(.+?)\s+nDCG\s+[\d.]+\s+vs ignore\s+([-+][\d.]+) \[\s*([-+][\d.]+),\s*([-+][\d.]+)\]\s+rel kept\s+([\d.]+)%\s+active\s+([\d.]+)%", b, re.M):
                rows[m.group(1)] = dict(zip(("diff", "lo", "hi", "kept", "active"), map(float, m.group(2, 3, 4, 5, 6))))
            out[key][h.group(1, 2)] = rows
    m = re.search(r"\((\d+)/(\d+) of all ESCI sample queries\)", t); out["overlap"] = (int(m.group(1)), int(m.group(2)))
    return out

L, D72, B = depth(f"{R}/depth_qwen3b_lora.txt"), depth(f"{R}/depth_qwen72.txt"), depth(f"{R}/depth_qwen3b_base.txt")
C4, C6 = depth(f"{R}/depth_qwen3b_lora_cut0.4.txt"), depth(f"{R}/depth_qwen3b_lora_cut0.6.txt")
LL = loss(f"{R}/loss_qwen3b_lora.txt"); RB = robust(f"{R}/robust_qwen3b_lora.txt"); P = {e: policy(f"{R}/policy_{e}.txt") for e in ("qwen3b_lora", "qwen72", "qwen3b_base")}
G, RU, ST, T1, T2 = J("grid"), J("rules"), J("stats"), J("table1"), J("table2")
RV = reviews(f"{R}/reviews.txt")
RUNS, SM, SH = G["runs"], G["summary"], G["shared"]
def rn(e, d, b="hybrid"): return RUNS[f"{e}|{d}|{b}"]
def cell(e, d, t): return next(c for c in RU["cells"] if (c["extractor"], c["retailer"], c["type"]) == (e, d, t))
S = "qwen3b_lora"; S7 = "qwen72"; S3 = "qwen3b_base"

# --- abstract (line 29), repeated in contributions (43, 44, 46)
for ln in (29, 43):
    add(ln, "hard-filter drop, grid minimum (18 runs)", "44", SM["drop_min"]); add(ln, "hard-filter drop, grid maximum", "64", SM["drop_max"])
add(29, "ESCI ignore nDCG@10", "85.0", L["esci"]["ignore"]["mean"]); add(29, "ESCI hard nDCG@10", "30.9", L["esci"]["hard filter"]["mean"])
add(29, "WANDS ignore nDCG@10", "75.8", L["wands"]["ignore"]["mean"]); add(29, "WANDS hard nDCG@10", "14.0", L["wands"]["hard filter"]["mean"])
claim(29, "p<0.001 for both hard-filter drops", max(L[d]["hard filter"]["p"] for d in L) < 0.001, max(L[d]["hard filter"]["p"] for d in L))
add(29, "LoRA F1", "0.73", T1[S]["f1"], prov="G"); add(29, "72B F1", "0.47", T1[S7]["f1"], prov="G")
add(29, "shared-query reduction, low end (ESCI)", "4", SH["esci"]["qwen72_drop"] - SH["esci"]["lora_drop"]); add(29, "shared-query reduction, high end (WANDS)", "6", SH["wands"]["qwen72_drop"] - SH["wands"]["lora_drop"])
add(29, "Spearman where a field exists", "0.91", RU["spearman_field"]); add(29, "rule correct of 38", 36, RU["rule"]["0.5"]["correct"], 0); add(29, "cells", 38, RU["n_cells"], 0)
add(121, "rule correct where a field exists", 20, RU["rule"]["0.5"]["correct_field"], 0); add(29, "cells with a field", 22, RU["n_field"], 0)
for ln in (29, 46):
    add(ln, "gating + title, loss avoided, grid minimum", "78", SM["avoid_gate_title_min"]); add(ln, "gating + title, loss avoided, grid maximum", "98", SM["avoid_gate_title_max"])
add(29, "oracle gain, hybrid, maximum", "1.5", SM["oracle_max_hybrid"]); add(29, "oracle gain, all rankers, maximum", "2.7", SM["oracle_max"])
wording = [v["wording"] for d in LL.values() for v in d.values() if v["kept"] + v["wording"] + v["disagree"] > 0]
for ln in (44, 89):
    add(ln, "wording share by type, minimum (types with a field)", "1.5", min(wording)); add(ln, "wording share by type, maximum", "10.2", max(wording))
# --- setup (55, 57, Table I, 74)
OC, OG, OR = J("esci_official/join_check"), J("esci_official/grid"), J("esci_official/rules")
add(55, "mirror repeats, share of rows", "2", 100 * (OC["sample_rows_mirror"] - OC["sample_rows_locale_join"]) / OC["sample_rows_mirror"]); add(55, "distinct products per query (official)", "19.0", OC["official_mean_rows"])
claim(55, "product-ID-only join reproduces the mirror rows exactly", OC["mirror_rows_not_in_locale_less_join"] == 0 and OC["locale_less_rows_not_in_mirror"] == 0, OC["mirror_rows_in_locale_less_join"])
claim(55, "ESCI hard-filter drop changes by at most 0.2 on official files", max(abs(G["runs"][k]["hard"]["diff"] - OG["runs"][k]["hard"]["diff"]) for k in G["runs"] if "|esci|" in k) <= 0.2, round(max(abs(G["runs"][k]["hard"]["diff"] - OG["runs"][k]["hard"]["diff"]) for k in G["runs"] if "|esci|" in k), 3))
claim(55, "no rule count changes on official files", all(RU["rule"][c]["correct"] == OR["rule"][c]["correct"] for c in RU["rule"]), [OR["rule"][c]["correct"] for c in ("0.4", "0.5", "0.6")])
add(55, "ESCI queries", 4000, ST["esci_queries"], 0); add(55, "ESCI mean judged", "19.4", ST["esci_mean_judged"]); add(55, "WANDS queries", 480, ST["wands_queries"], 0)
add(55, "WANDS products", 42994, ST["wands_products"], 0); add(55, "WANDS judgments", 233448, ST["wands_judgments"], 0); add(55, "WANDS judgments per query", "486", ST["wands_judgments_per_query"])
add(55, "WANDS feature keys (lower-cased, non-empty value)", 7949, ST["wands_feature_keys"], 0)
claim(55, "ESCI gains 1 / 0.1 / 0.01 / 0 in build_index.py", code("src/build_index.py", r'"Exact": 1\.0, "Substitute": 0\.1, "Complement": 0\.01, "Irrelevant": 0\.0'), "code")
claim(55, "WANDS gains 2 / 1 in build_index.py", code("src/build_index.py", r'"Exact": 2, "Partial": 1'), "code")
add(57, "QueryNER train", 7841, ST["queryner_train"], 0); add(57, "training overlap share (normalized text match, reviews.py)", "8.0", 100 * RV["overlap"][0] / RV["overlap"][1])
claim(57, "overlap: 318 normalized matches (stats.py exact match finds 316)", RV["overlap"][0] == 318 and ST["overlap_queryner_train"] == 316, f"{RV['overlap'][0]} / {ST['overlap_queryner_train']}")
add(57, "LoRA ESCI drop, all queries", "-54.1", RV["pol"][(S, "esci")]["hard filter"]["diff"]); add(57, "LoRA ESCI drop, overlap excluded", "-53.7", RV["polx"][(S, "esci")]["hard filter"]["diff"])
add(57, "gated result change, overlap excluded", "0.3", abs(RV["polx"][(S, "esci")]["gate:safety + fallback"]["diff"] - RV["pol"][(S, "esci")]["gate:safety + fallback"]["diff"]))
claim(57, "LoRA rank 16, 2 epochs in lora_train.py", code("gpu/lora_train.py", r"r=16") and code("gpu/lora_train.py", r"B, EP = 32, 2"), "code", "G")
claim(57, "five samples, temperature 0.7 in extract.py", code("gpu/extract.py", r"sampling\(SamplingParams, 5, 0\.7\)"), "code", "G")
add(61, "QueryNER test queries", 993, ST["queryner_test"], 0)
for e, p, r, f in ((S7, "0.785", "0.337", "0.472"), (S3, "0.592", "0.265", "0.366"), (S, "0.748", "0.714", "0.731")):
    ln = {S7: 67, S3: 68, S: 69}[e]; add(ln, f"Table I {e} precision", p, T1[e]["precision"], prov="G"); add(ln, f"Table I {e} recall", r, T1[e]["recall"], prov="G"); add(ln, f"Table I {e} F1", f, T1[e]["f1"], prov="G")
claim(74, "top 300 kept on WANDS in build_index.py", code("src/build_index.py", r"\[:300\]"), "code")
add(74, "evaluated ESCI queries", 1699, L["esci"]["n"], 0); add(74, "evaluated WANDS queries", 255, L["wands"]["n"], 0)
t = open("provenance/gpu_timing.txt").read() if os.path.exists("provenance/gpu_timing.txt") else None
m = re.search(r"elapsed_seconds=(\d+)", t) if t else None
claim(74, "GPU work under one hour (saved timing log)", bool(m) and int(m.group(1)) < 3600, m.group(1) if m else None, "G")
# --- cost (78)
add(78, "ESCI drop", "-54.1", L["esci"]["hard filter"]["diff"]); add(78, "ESCI drop CI low", "-56.0", L["esci"]["hard filter"]["dlo"], STOCH); add(78, "ESCI drop CI high", "-52.0", L["esci"]["hard filter"]["dhi"], STOCH)
add(78, "WANDS drop", "-61.8", L["wands"]["hard filter"]["diff"]); add(78, "WANDS drop CI low", "-66.1", L["wands"]["hard filter"]["dlo"], STOCH); add(78, "WANDS drop CI high", "-57.4", L["wands"]["hard filter"]["dhi"], STOCH)
add(78, "72B ESCI drop", "-63.4", D72["esci"]["hard filter"]["diff"]); add(78, "72B WANDS drop", "-51.5", D72["wands"]["hard filter"]["diff"])
h72 = [rn(S7, d, b)["hard"]["mean"] for d in ("esci", "wands") for b in ("bm25", "dense")]; hL = [rn(S, d, b)["hard"]["mean"] for d in ("esci", "wands") for b in ("bm25", "dense")]
add(78, "72B BM25/dense hard, low", "16", min(h72)); add(78, "72B BM25/dense hard, high", "20", max(h72)); add(78, "LoRA BM25/dense hard, low", "13", min(hL)); add(78, "LoRA BM25/dense hard, high", "31", max(hL))
add(78, "3B prompted ESCI hard", "20.6", B["esci"]["hard filter"]["mean"]); add(78, "3B prompted WANDS hard", "11.9", B["wands"]["hard filter"]["mean"])
rx = [rn(e, "wands")["relax"] for e in (S, S7, S3)]; rxe = [rn(e, "esci")["relax"]["diff"] for e in (S, S7, S3)]
add(78, "relaxation cost on WANDS, low", "2.9", min(-x["diff"] for x in rx)); add(78, "relaxation cost on WANDS, high", "5.1", max(-x["diff"] for x in rx))
claim(78, "relaxation p<0.002 on WANDS", max(x["p"] for x in rx) < 0.002, max(x["p"] for x in rx)); claim(78, "relaxation recovers the loss on ESCI (no significant cost)", min(rxe) >= 0, round(min(rxe), 2))
claim(83, "Fig 1 is LoRA + hybrid (n in panel titles)", L["esci"]["n"] == 1699 and L["wands"]["n"] == 255, "n")
# --- why (89, 98)
add(89, "ESCI disagree, low (color)", "31", LL["esci"]["color"]["disagree"]); add(89, "ESCI disagree, high (brand)", "38", LL["esci"]["brand"]["disagree"])
add(89, "WANDS color disagree", "43", LL["wands"]["color"]["disagree"]); add(89, "WANDS color empty", "36", LL["wands"]["color"]["empty"])
claim(89, "ESCI size/material/shape and WANDS brand 100% empty", all(LL["esci"][t]["empty"] == 100 for t in ("size", "material", "shape")) and LL["wands"]["brand"]["empty"] == 100, "100%")
claim(89, "wording below 10% at every cosine threshold", max(RB[d][k] for d in RB for k in ("w0.7", "w0.8", "w0.9")) < 10, max(RB[d][k] for d in RB for k in ("w0.7", "w0.8", "w0.9")))
add(98, "shared ESCI 72B drop", "62.8", SH["esci"]["qwen72_drop"]); add(98, "shared ESCI LoRA drop", "58.4", SH["esci"]["lora_drop"]); add(98, "shared ESCI n", 659, SH["esci"]["n"], 0)
add(98, "shared WANDS 72B drop", "53.9", SH["wands"]["qwen72_drop"]); add(98, "shared WANDS LoRA drop", "48.3", SH["wands"]["lora_drop"]); add(98, "shared WANDS n", 118, SH["wands"]["n"], 0)
add(98, "attribute ratio ESCI ('about twice')", 2, ST["attr_ratio_lora_vs_72b"]["esci"], 0.1); add(98, "attribute ratio WANDS", "1.6", ST["attr_ratio_lora_vs_72b"]["wands"])
claim(98, "spa filter title does not name the queried brand", not T2["spa_title_names_brand"], True)
for i, r in enumerate(T2["rows"]): claim(108 + i, f"Table II row {i + 1}: {r['query']} / {r['attribute']}", r["ok"], r["ok"])
# --- safety (119, 121, 123, Table III, Fig 3)
claim(119, "K=50 in safety_check.py", code("src/safety_check.py", r"\[:50\]"), "code")
add(121, "Spearman all cells", "0.98", RU["spearman_all"]); add(121, "Spearman where a field exists", "0.91", RU["spearman_field"]); add(121, "Spearman within ESCI", "0.66", RU["spearman_field_esci"]); add(121, "n within ESCI", 6, RU["n_field_esci"], 0)
add(121, "Spearman within WANDS", "0.82", RU["spearman_field_wands"]); add(121, "n within WANDS", 16, RU["n_field_wands"], 0)
ms = sorted(tuple(m[:3]) for m in RU["rule"]["0.5"]["misses"]); claim(121, "the two errors are 3B prompted WANDS color and material", ms == [(S3, "wands", "color"), (S3, "wands", "material")], str(ms))
c3, m3 = cell(S3, "wands", "color"), cell(S3, "wands", "material")
add(121, "3B WANDS color safety", "0.51", c3["safety"]); add(121, "3B WANDS color kept", "0.16", c3["kept"]); add(121, "3B WANDS material safety", "0.53", m3["safety"]); add(121, "3B WANDS material kept", "0.495", m3["kept"])
wc = [cell(e, "wands", "color") for e in (S, S7, S3)]; add(121, "WANDS color max safety", "0.51", max(c["safety"] for c in wc)); add(121, "WANDS color kept, low", "16", 100 * min(c["kept"] for c in wc)); add(121, "WANDS color kept, high", "17", 100 * max(c["kept"] for c in wc))
add(121, "lenient share ESCI", "4.0", RB["esci"]["lenient"]); add(121, "lenient share WANDS", "22.3", RB["wands"]["lenient"]); add(207, "lenient share ESCI (rounded)", "4", RB["esci"]["lenient"]); add(207, "lenient share WANDS (rounded)", "22", RB["wands"]["lenient"])
for cut, e in (("0.4", 34), ("0.5", 36), ("0.6", 38)):
    add(123, f"rule correct at {cut}", e, RU["rule"][cut]["correct"], 0); add(169, f"Table IV rule correct at {cut}", e, RU["rule"][cut]["correct"], 0)
for ln in (123, 207): add(ln, "largest change in drop, lenient out", "2.2", max(RB[d]["drop_all"] - RB[d]["drop_len"] for d in RB))
for ln, d, t, cv, cs, sf, kp in ((135, "esci", "brand", "0.95", "0.78", "0.74", "0.55"), (136, "esci", "color", "0.90", "0.76", "0.69", "0.56"), (138, "wands", "material", "0.96", "0.75", "0.72", "0.60"),
                                 (139, "wands", "color", "0.74", "0.58", "0.43", "0.17"), (140, "wands", "shape", "0.50", "0.17", "0.08", "0.10"), (141, "wands", "size", "0.25", "0.32", "0.08", "0.03")):
    c = cell(S, d, t)
    add(ln, f"Table III {d} {t} cover", cv, c["coverage"]); add(ln, f"Table III {d} {t} consist.", cs, c["consistency"]); add(ln, f"Table III {d} {t} safety", sf, c["safety"]); add(ln, f"Table III {d} {t} kept", kp, c["kept"])
for ln, d, t in ((137, "esci", "size"), (137, "esci", "material"), (137, "esci", "shape"), (142, "wands", "brand")):
    c = cell(S, d, t); claim(ln, f"Table III {d} {t}: cover, safety, kept all 0.00", c["coverage"] == 0 and c["safety"] == 0 and c["kept"] == 0, 0)
add(150, "Fig 3 cells", 38, RU["n_cells"], 0); add(150, "Fig 3 circled (misclassified) cells", 2, len(RU["rule"]["0.5"]["misses"]), 0)
# --- Table IV (164-176)
for d, ln, vals in (("esci", 164, ("4.4", "1.6", "0.3")), ("wands", 165, ("9.4", "2.6", "0.4"))):
    for tau, v in zip(("0.7", "0.8", "0.9"), vals): add(ln, f"Table IV wording {d} tau={tau}", v, RB[d]["w" + tau])
add(170, "Table IV loss avoided ESCI cut 0.4", "79", avoid(C4["esci"])); add(170, "Table IV loss avoided ESCI cut 0.5", "79", avoid(L["esci"])); add(170, "Table IV loss avoided ESCI cut 0.6", "79", avoid(C6["esci"]))
add(171, "Table IV loss avoided WANDS cut 0.4", "88", avoid(C4["wands"])); add(171, "Table IV loss avoided WANDS cut 0.5", "97", avoid(L["wands"])); add(171, "Table IV loss avoided WANDS cut 0.6", "97", avoid(C6["wands"]))
add(175, "Table IV ESCI drop", "54.1", RB["esci"]["drop_all"]); add(175, "Table IV ESCI lenient-out drop", "53.6", RB["esci"]["drop_len"]); add(175, "Table IV ESCI share lenient", "4.0", RB["esci"]["lenient"])
add(176, "Table IV WANDS drop", "61.8", RB["wands"]["drop_all"]); add(176, "Table IV WANDS lenient-out drop", "59.6", RB["wands"]["drop_len"]); add(176, "Table IV WANDS share lenient", "22.3", RB["wands"]["lenient"])
# --- mitigation (183, Table V 194-199, 204)
add(183, "title fallback avoided ESCI", "58", avoid(L["esci"], "hard filter + title fallback (repair)")); add(183, "title fallback avoided WANDS", "40", avoid(L["wands"], "hard filter + title fallback (repair)"))
add(183, "gating + title avoided ESCI, LoRA", "79", avoid(L["esci"])); add(183, "gating + title avoided WANDS, LoRA", "97", avoid(L["wands"]))
add(183, "gating + title avoided ESCI, 72B", "89", avoid(D72["esci"])); add(183, "gating + title avoided WANDS, 72B", "94", avoid(D72["wands"]))
gaps = [-x[d]["safety-gated + title fallback"]["diff"] for x in (L, D72) for d in x]
add(183, "gap to unfiltered, low", "2.2", min(gaps)); add(183, "gap to unfiltered, high", "11.4", max(gaps))
claim(204, "oracle searches subsets of up to four attributes in depth.py", code("src/depth.py", r"if len\(at\) <= 4"), "code")
add(204, "oracle ESCI", "0.9", L["esci"]["oracle subset of hard filters"]["diff"]); add(204, "oracle ESCI CI low", "0.7", L["esci"]["oracle subset of hard filters"]["dlo"], STOCH); add(204, "oracle ESCI CI high", "1.1", L["esci"]["oracle subset of hard filters"]["dhi"], STOCH)
add(204, "oracle WANDS", "0.4", L["wands"]["oracle subset of hard filters"]["diff"]); add(204, "oracle WANDS CI low", "0.1", L["wands"]["oracle subset of hard filters"]["dlo"], STOCH); add(204, "oracle WANDS CI high", "0.8", L["wands"]["oracle subset of hard filters"]["dhi"], STOCH)
add(204, "oracle ESCI, 72B", "1.0", D72["esci"]["oracle subset of hard filters"]["diff"]); add(204, "oracle WANDS, 72B", "1.5", D72["wands"]["oracle subset of hard filters"]["diff"])
pol = [v for e in P.values() for v in e.values()]
claim(204, "learned policy, other retailer: no meaningful gain over ignoring (at most +0.5 in 6 runs)", max(pol) <= 0.5, round(max(pol), 2))
# --- limitations (207)
ns = [rn(e, "wands")["n"] for e in (S, S7, S3)]; add(207, "WANDS slice, low", 123, min(ns), 0); add(207, "WANDS slice, high", 255, max(ns), 0)

# --- camera-ready additions (v10/v11): reviewer analyses in results/reviews.txt
sup = [c["kept"] for c in RU["cells"] if c["kept"] >= 0.5]
add(29, "supported cells keep, low", "50", 100 * min(sup)); add(29, "supported cells keep, high", "68", 100 * max(sup))
add(29, "WANDS color score (abstract)", "0.51", max(c["safety"] for c in wc)); add(29, "WANDS color kept (abstract)", "16", 100 * c3["kept"])
add(29, "rule correct of 38 (abstract)", 36, RU["rule"]["0.5"]["correct"], 0)
claim(29, "relaxation keeps a filter on about a quarter of ESCI queries", 20 <= RV["pol"][(S, "esci")]["fixed-order relaxation"]["active"] <= 30, RV["pol"][(S, "esci")]["fixed-order relaxation"]["active"])
claim(119, "consistency set to 0 below ten title matches (rules.py, safety_check.py, reviews.py)", all(code(f, r'len\((A|X)\["cons"\]\) >= 10 else 0\.0') for f in ("src/rules.py", "src/safety_check.py", "src/reviews.py")), "code")
claim(119, "no cell keeps 80%", max(c["kept"] for c in RU["cells"]) < 0.8, round(max(c["kept"] for c in RU["cells"]), 3)); add(119, "best cell keeps", "68", 100 * max(c["kept"] for c in RU["cells"]))
for name, k, fa in (("exists", 8, 14), ("coverage", 15, 7), ("agreement", 18, 4), ("safety", 20, 2)):
    add(121, f"{name} alone correct of 22", k, RV["abl"][name]["correct"], 0); add(121, f"{name} false approvals", fa, RV["abl"][name]["fa"], 0)
H = RV["heldout"]
add(121, "tuned cut-off mean", "0.56", H["tau"][0]); add(121, "tuned cut-off 2.5%", "0.45", H["tau"][1]); add(121, "tuned cut-off 97.5%", "0.65", H["tau"][2])
add(121, "held-out accuracy, tuned", "93", 100 * H["acc_tuned"][0]); add(121, "held-out accuracy, 0.5 fixed", "92", 100 * H["acc_fixed"][0]); add(121, "held-out accuracy, field cells, 0.5 fixed", "86", 100 * H["acc_fixed_field"][0])
for ln, d, t, q, pr in ((135, "esci", "brand", 903, 9700), (136, "esci", "color", 265, 3211), (138, "wands", "material", 55, 5771), (139, "wands", "color", 57, 7141),
                        (140, "wands", "shape", 47, 5068), (141, "wands", "size", 45, 3054), (142, "wands", "brand", 98, 13386)):
    add(ln, f"Table III {d} {t} queries", q, RV["cells"][(S, d, t)]["q"], 0); add(ln, f"Table III {d} {t} pairs", pr, RV["cells"][(S, d, t)]["pairs"], 0)
z = [RV["cells"][(S, "esci", t)] for t in ("size", "material", "shape")]
add(137, "Table III ESCI size/mat/shape queries, low", 186, min(c["q"] for c in z), 0); add(137, "Table III ESCI size/mat/shape queries, high", 379, max(c["q"] for c in z), 0)
add(137, "Table III ESCI size/mat/shape pairs", 10382, sum(c["pairs"] for c in z), 0)
PO = lambda e, d, k: RV["pol"][(e, d)][k]
add(183, "WANDS gate LoRA", "-2.2", PO(S, "wands", "gate:safety + fallback")["diff"]); add(183, "WANDS gate 72B", "-3.2", PO(S7, "wands", "gate:safety + fallback")["diff"])
add(183, "WANDS relax LoRA", "-2.9", PO(S, "wands", "fixed-order relaxation")["diff"]); add(183, "WANDS relax 72B", "-5.1", PO(S7, "wands", "fixed-order relaxation")["diff"])
add(183, "WANDS gate 3B prompted", "-12.8", PO(S3, "wands", "gate:safety + fallback")["diff"]); add(183, "WANDS relax 3B prompted", "-5.0", PO(S3, "wands", "fixed-order relaxation")["diff"])
claim(183, "gate and relaxation intervals overlap (LoRA, 72B on WANDS)", all(PO(e, "wands", "gate:safety + fallback")["lo"] <= PO(e, "wands", "fixed-order relaxation")["hi"] and PO(e, "wands", "fixed-order relaxation")["lo"] <= PO(e, "wands", "gate:safety + fallback")["hi"] for e in (S, S7)), "CI")
xs = {e: [PO(e, "wands", "gate:safety + fallback")["diff"] - PO(e, "wands", k)["diff"] for k in ("gate:coverage + fallback", "gate:agreement + fallback")] for e in (S, S7)}
add(183, "coverage/consistency alone, extra cost LoRA", "5", min(xs[S])); add(183, "coverage/consistency alone, extra cost 72B", "13", max(xs[S7]))
claim(183, "ESCI: relaxation active on a quarter, gating on two-thirds", 20 <= PO(S, "esci", "fixed-order relaxation")["active"] <= 30 and 62 <= PO(S, "esci", "gate:safety + fallback")["active"] <= 70,
      (PO(S, "esci", "fixed-order relaxation")["active"], PO(S, "esci", "gate:safety + fallback")["active"]))
add(183, "ESCI gate cost (text)", "11.4", -PO(S, "esci", "gate:safety + fallback")["diff"])
TV = ((194, "hard filter", ("-54.1", "30", "100"), ("-61.8", "13", "100")), (195, "fixed-order relaxation", ("+0.2", "98", "26"), ("-2.9", "89", "27")),
      (196, "gate:exists + fallback", ("-11.4", "80", "67"), ("-16.6", "58", "71")), (197, "gate:coverage + fallback", ("-11.4", "80", "67"), ("-7.1", "79", "41")),
      (198, "gate:agreement + fallback", ("-11.4", "80", "67"), ("-7.1", "79", "41")), (199, "gate:safety + fallback", ("-11.4", "80", "67"), ("-2.2", "91", "22")))
for ln, k, *vals in TV:
    for d, (dv, kp, ac) in zip(("esci", "wands"), vals):
        r = PO(S, d, k); add(ln, f"Table V {d} {k} dnDCG", dv, r["diff"]); add(ln, f"Table V {d} {k} kept", kp, r["kept"]); add(ln, f"Table V {d} {k} active", ac, r["active"])
add(191, "Table V ESCI n", 1699, L["esci"]["n"], 0); add(191, "Table V WANDS n", 255, L["wands"]["n"], 0)

if len(sys.argv) > 1:  # optional: confirm each expected value is printed on its line of the paper source
    tex = open(sys.argv[1]).read().split("\n"); off = []
    for ln, what, exp, *_ in ROWS:
        if exp == "true": continue
        v = str(exp).lstrip("-"); v2 = f"{int(float(v)):,}".replace(",", "{,}") if v.isdigit() else v
        if v not in tex[ln - 1] and v2 not in tex[ln - 1]: off.append((ln, what, exp))
    print("expected values not found on their paper line:", len(off))
    for o in off: print("  ", o)
n = {k: sum(r[-1] == k for r in ROWS) for k in ("PASS", "FAIL", "MISSING")}
print(f"{'line':>4} {'status':7} {'prov':4} {'expected':>9} {'actual':>10} {'tol':>6}  claim")
for ln, what, exp, act, tol, prov, st in ROWS: print(f"{ln:>4} {st:7} {prov:4} {str(exp):>9} {str(act):>10} {str(tol):>6}  {what}")
print(f"\nTOTAL {len(ROWS)}: PASS {n['PASS']}  FAIL {n['FAIL']}  MISSING {n['MISSING']}")
for ln, what, exp, act, tol, prov, st in ROWS:
    if st != "PASS": print(f"{st} line {ln}: {what}: expected {exp}, actual {act}")
sys.exit(1 if n["FAIL"] or n["MISSING"] else 0)
