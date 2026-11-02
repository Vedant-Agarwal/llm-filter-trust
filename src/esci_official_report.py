"""Mirror (paper) vs official ESCI, from results/ and results/esci_official/. Not used for the paper's numbers."""
import json, re, sys
J = lambda p: json.load(open(p)); A, B = J("results/grid.json"), J("results/esci_official/grid.json"); RA, RB = J("results/rules.json"), J("results/esci_official/rules.json"); C = J("results/esci_official/join_check.json")
print(f"rows per query: mirror {C['sample_rows_mirror'] / 4000:.2f}, official {C['official_mean_rows']:.2f}, minimum official {C['official_min_rows']}")
print(f"mirror rows found in a product_id-only join: {C['mirror_rows_in_locale_less_join']} of {C['sample_rows_mirror']}; join rows not in mirror: {C['locale_less_rows_not_in_mirror']}")
print(f"official US test queries {C['official_us_test_queries']}, with >=8 rows {C['official_us_test_queries_with_8']}; same rule on official rows selects {C['fresh_sample_same_rule_overlap']} of the same 4,000")
print("ESCI, by extractor and ranker: mirror -> official")
for k in A["runs"]:
    if "|esci|" not in k: continue
    a, b = A["runs"][k], B["runs"][k]
    print(f"{k:24s} n {a['n']}->{b['n']} ignore {a['ignore']:.1f}->{b['ignore']:.1f} hard {a['hard']['mean']:.1f}->{b['hard']['mean']:.1f} drop {-a['hard']['diff']:.1f}->{-b['hard']['diff']:.1f} title {a['avoid_title']:.1f}->{b['avoid_title']:.1f}% gate+title {a['avoid_gate_title']:.1f}->{b['avoid_gate_title']:.1f}% relax {a['relax']['diff']:+.2f}->{b['relax']['diff']:+.2f} oracle {a['oracle']['diff']:+.2f}->{b['oracle']['diff']:+.2f}")
for k in A["summary"]: print(f"grid {k}: {A['summary'][k]:.2f} -> {B['summary'][k]:.2f}")
print("shared esci:", {k: round(v, 2) for k, v in A["shared"]["esci"].items()}, "->", {k: round(v, 2) for k, v in B["shared"]["esci"].items()})
for k in ("spearman_all", "spearman_field", "spearman_field_esci", "spearman_field_wands"): print(f"{k}: {RA[k]:.3f} -> {RB[k]:.3f}")
print("rule correct at 0.4/0.5/0.6:", [RA["rule"][c]["correct"] for c in ("0.4", "0.5", "0.6")], "->", [RB["rule"][c]["correct"] for c in ("0.4", "0.5", "0.6")], "misses at 0.5:", RB["rule"]["0.5"]["misses"])
for e in ("qwen3b_lora",):
    for x, y in zip([c for c in RA["cells"] if c["extractor"] == e and c["retailer"] == "esci" and c["coverage"] > 0], [c for c in RB["cells"] if c["extractor"] == e and c["retailer"] == "esci" and c["coverage"] > 0]):
        print(f"{e} ESCI {x['type']}: cover {x['coverage']:.3f}->{y['coverage']:.3f} consist {x['consistency']:.3f}->{y['consistency']:.3f} safety {x['safety']:.3f}->{y['safety']:.3f} kept {x['kept']:.3f}->{y['kept']:.3f}")
