"""Check the ESCI mirror against Amazon's official parquet files (https://github.com/amazon-science/esci-data).
Keeps the paper's 4,000 queries; joins examples to products on (product_id, product_locale), and also on product_id alone to test
whether that reproduces the mirror's rows. Writes $ISIR_DATA/official/esci_test_us_4000.jsonl in the format build_index.py reads.
Needs $ISIR_OFFICIAL with shopping_queries_dataset_{examples,products}.parquet and $ISIR_DATA/esci_test_us_4000.jsonl."""
import collections, json, os, random, re
import pandas as pd
DATA = os.environ.get("ISIR_DATA", "data"); OFF = os.environ["ISIR_OFFICIAL"]; OUT = os.environ.get("ISIR_RESULTS", "results/esci_official")
LAB = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}
cjk = lambda s: bool(re.search(r"[぀-ヿ一-鿿]", str(s or "")))
ex = pd.read_parquet(f"{OFF}/shopping_queries_dataset_examples.parquet"); pr = pd.read_parquet(f"{OFF}/shopping_queries_dataset_products.parquet")
us = ex[(ex.product_locale == "us") & (ex.split == "test")]
out = {"official_us_test_rows": len(us), "official_us_test_queries": int(us.query_id.nunique()), "official_duplicate_query_product": int(us.duplicated(["query_id", "product_id"]).sum()),
       "products_rows": len(pr), "products_unique_id": int(pr.product_id.nunique()), "products_unique_id_locale": int(pr.drop_duplicates(["product_id", "product_locale"]).shape[0]),
       "products_ids_in_several_locales": int((pr.groupby("product_id").product_locale.nunique() > 1).sum())}
M = [json.loads(l) for l in open(f"{DATA}/esci_test_us_4000.jsonl")]; ids = {r["qid"] for r in M}
right = us[us.query_id.isin(ids)].merge(pr, on=["product_id", "product_locale"], how="left", validate="many_to_one")
naive = us[us.query_id.isin(ids)].merge(pr.drop(columns="product_locale"), on="product_id", how="left")
out.update(sample_rows_locale_join=len(right), sample_rows_locale_less_join=len(naive), sample_rows_mirror=sum(len(r["cands"]) for r in M), unmatched_products_locale_join=int(right.product_title.isna().sum()))
# does a locale-less join reproduce the mirror exactly? compare multisets of (query, product, label, title, brand, color, truncated bullets)
key = lambda q, p, l, t, b, c, bu: (int(q), p, l, t, b, c, (bu or "")[:600])
none = lambda v: None if pd.isna(v) else v
mir = collections.Counter(key(r["qid"], c["pid"], c["label"], c["title"], c["brand"], c["color"], c["bullets"]) for r in M for c in r["cands"])
loc = collections.Counter(key(r.query_id, r.product_id, LAB[r.esci_label], none(r.product_title), none(r.product_brand), none(r.product_color), none(r.product_bullet_point)) for r in naive.itertuples())
rig = collections.Counter(key(r.query_id, r.product_id, LAB[r.esci_label], none(r.product_title), none(r.product_brand), none(r.product_color), none(r.product_bullet_point)) for r in right.itertuples())
out.update(mirror_rows_in_locale_less_join=sum((mir & loc).values()), mirror_rows_not_in_locale_less_join=sum((mir - loc).values()), locale_less_rows_not_in_mirror=sum((loc - mir).values()),
           mirror_rows_in_locale_join=sum((mir & rig).values()), mirror_rows_not_in_locale_join=sum((mir - rig).values()), locale_join_rows_not_in_mirror=sum((rig - mir).values()))
# same labels on the same (query, product)?
lm = {(r["qid"], c["pid"]): c["label"] for r in M for c in r["cands"]}; lo = {(int(r.query_id), r.product_id): LAB[r.esci_label] for r in right.itertuples()}
out.update(label_pairs_mirror=len(lm), label_pairs_official=len(lo), label_mismatches=sum(lm.get(k) != v for k, v in lo.items()))
# the sample rule on official rows
n = right.groupby("query_id").size(); out.update(official_mean_rows=float(n.mean()), official_min_rows=int(n.min()), sample_queries_found=int(len(n)), sample_queries_below_8_official=int((n < 8).sum()))
qs = us.groupby(["query_id", "query"]).size(); keys = sorted(qs.index.tolist()); random.Random(0).shuffle(keys)
fresh = [k for k in keys if qs[k] >= 8][:4000]
out.update(fresh_sample_same_rule_overlap=len({k[0] for k in fresh} & ids), official_us_test_queries_with_8=int((qs >= 8).sum()))
# the 'wd' example and Japanese text
wd = [q for q in M if re.search(r"\bwd\b", q["query"])]
rows = {}
for q in wd:
    a = right[right.query_id == q["qid"]]
    rows[q["query"]] = {"mirror_japanese_brand_rows": sum(cjk(c["brand"]) for c in q["cands"]), "official_japanese_brand_rows": int(a.product_brand.map(cjk).sum()), "official_rows": len(a), "mirror_rows": len(q["cands"]),
                        "official_exact_brands": sorted({str(b) for b in a[a.esci_label == "E"].product_brand})[:6]}
out.update(wd_queries=rows, official_cjk_title_rows=int(right.product_title.map(cjk).sum()), mirror_cjk_title_rows=sum(cjk(c["title"]) for r in M for c in r["cands"]))
# candidates for build_index.py, same 4,000 queries in the same order
by = collections.defaultdict(list)
for r in right.itertuples(): by[int(r.query_id)].append({"pid": r.product_id, "label": LAB[r.esci_label], "title": none(r.product_title), "brand": none(r.product_brand), "color": none(r.product_color), "bullets": (none(r.product_bullet_point) or "")[:600]})
os.makedirs(f"{DATA}/official", exist_ok=True); os.makedirs(OUT, exist_ok=True)
with open(f"{DATA}/official/esci_test_us_4000.jsonl", "w") as f:
    for r in M: f.write(json.dumps({"qid": r["qid"], "query": r["query"], "cands": by[r["qid"]]}) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False, default=float)); json.dump(out, open(f"{OUT}/join_check.json", "w"), indent=1, ensure_ascii=False, default=float)
