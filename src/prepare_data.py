"""Build the ESCI query sample and the query lists fed to the extractors.
Sample: US-locale test queries with at least 8 judged products, shuffled with random.Random(0), first 4,000.
Source: tasksource/esci on Hugging Face, the mirror the paper's run used, pinned to the revision below.
Writes to $ISIR_DATA: esci_test_us_4000.jsonl, esci_queries.jsonl, wands_queries.jsonl.  Prints SHA-256 of each."""
import collections, datasets, hashlib, json, os, pandas as pd, random
DATA = os.environ.get("ISIR_DATA", "data"); WANDS = os.environ.get("ISIR_WANDS", "WANDS")
REV = "8113b17a5d4099e20243282c926f1bc1a08a4d13"
os.makedirs(DATA, exist_ok=True)
ds = datasets.load_dataset("tasksource/esci", split="test", revision=REV)
ds = ds.filter(lambda r: r["product_locale"] == "us")
byq = collections.defaultdict(list)
for r in ds:
    byq[(r["query_id"], r["query"])].append({"pid": r["product_id"], "label": r["esci_label"], "title": r["product_title"], "brand": r["product_brand"],
                                             "color": r["product_color"], "bullets": (r["product_bullet_point"] or "")[:600]})
keys = sorted(byq); random.Random(0).shuffle(keys)
pick = [k for k in keys if len(byq[k]) >= 8][:4000]
with open(f"{DATA}/esci_test_us_4000.jsonl", "w") as f:
    for (qid, q) in pick: f.write(json.dumps({"qid": qid, "query": q, "cands": byq[(qid, q)]}) + "\n")
with open(f"{DATA}/esci_queries.jsonl", "w") as f:
    for (qid, q) in pick: f.write(json.dumps({"qid": f"esci_{qid}", "query": q}) + "\n")
wq = pd.read_csv(f"{WANDS}/dataset/query.csv", sep="\t")
with open(f"{DATA}/wands_queries.jsonl", "w") as f:
    for r in wq.itertuples(): f.write(json.dumps({"qid": f"wands_{r.query_id}", "query": r.query, "query_class": r.query_class}) + "\n")
print("ESCI US test queries total", len(byq), "| picked", len(pick), "| mean cands", round(sum(len(byq[k]) for k in pick) / len(pick), 1))
print("WANDS queries", len(wq))
for n in ("esci_test_us_4000", "esci_queries", "wands_queries"):
    print(n, hashlib.sha256(open(f"{DATA}/{n}.jsonl", "rb").read()).hexdigest())
