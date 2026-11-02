"""Build per-dataset candidate scoring (BM25 + dense bge-small, CPU) and catalog attribute fields.
Outputs $ISIR_DATA/<ds>_cands.pkl: list of {qid, query, cands:[{pid, gain, fields:{type: value or ''}, bm25, dense}]}.
WANDS: full catalog scored, keep top-300 by hybrid + all Exact/Partial judged (for recall accounting).
ESCI: judged candidates only (~19/query)."""
import json, os, pickle, re, sys, numpy as np, pandas as pd, bm25s
from sentence_transformers import SentenceTransformer
DATA = os.environ.get("ISIR_DATA", "data"); WANDS = os.environ.get("ISIR_WANDS", "WANDS")
TYPES = ["color", "brand", "material", "shape", "style", "size", "pattern"]
emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
QP = "Represent this sentence for searching relevant passages: "
def enc(texts, bs=256):
    return emb.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
def norm01(x):
    x = np.asarray(x, float); lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo + 1e-9)

def wands():
    p = pd.read_csv(f"{WANDS}/dataset/product.csv", sep="\t"); q = pd.read_csv(f"{WANDS}/dataset/query.csv", sep="\t")
    l = pd.read_csv(f"{WANDS}/dataset/label.csv", sep="\t")
    gain = {(r.query_id, r.product_id): {"Exact": 2, "Partial": 1}.get(r.label, 0) for r in l.itertuples()}
    def parse(s):
        d = {}
        for kv in str(s).split("|"):
            if ":" in kv:
                k, v = kv.split(":", 1); k = k.strip().lower(); v = v.strip()
                if v and v.lower() not in ("nan", "none", "n/a"): d.setdefault(k, []).append(v)
        return d
    F = p.product_features.map(parse)
    def field(d, t):
        ks = [k for k in d if t in k] if t != "brand" else []
        return " ; ".join(v for k in ks for v in d[k])
    fields = [{t: field(d, t) for t in TYPES} for d in F]
    text = (p.product_name.fillna("") + " . " + p.product_class.fillna("") + " . " + p.product_description.fillna("").str[:400]).tolist()
    ret = bm25s.BM25(); ret.index(bm25s.tokenize(text, stopwords="en"))
    D = enc(text); Qe = enc([QP + s for s in q["query"]])
    out = []
    for i, r in enumerate(q.itertuples()):
        docs, sc = ret.retrieve(bm25s.tokenize([r.query], stopwords="en"), k=len(text))
        bm = np.zeros(len(text)); bm[docs[0]] = sc[0]
        de = D @ Qe[i]
        h = 0.5 * norm01(bm) + 0.5 * norm01(de)
        keep = set(np.argsort(-h)[:300]) | {j for j, pid in enumerate(p.product_id) if gain.get((r.query_id, pid), 0) > 0}
        cands = [{"pid": int(p.product_id[j]), "gain": gain.get((r.query_id, p.product_id[j]), 0), "fields": fields[j],
                  "bm25": float(bm[j]), "dense": float(de[j]), "title": p.product_name[j]} for j in keep]
        out.append({"qid": f"wands_{r.query_id}", "query": r.query, "cands": cands})
    pickle.dump(out, open(f"{DATA}/wands_cands.pkl", "wb")); print("wands", len(out), "mean cands", np.mean([len(o["cands"]) for o in out]))

def esci():
    G = {"Exact": 1.0, "Substitute": 0.1, "Complement": 0.01, "Irrelevant": 0.0}
    rows = [json.loads(l) for l in open(f"{DATA}/esci_test_us_4000.jsonl")]
    allt = [c["title"] or "" for r in rows for c in r["cands"]]
    D = enc([(c["title"] or "") + " . " + (c["bullets"] or "")[:300] for r in rows for c in r["cands"]]); Qe = enc([QP + r["query"] for r in rows])
    out, k = [], 0
    for i, r in enumerate(rows):
        n = len(r["cands"]); texts = [(c["title"] or "") + " " + (c["bullets"] or "")[:300] for c in r["cands"]]
        ret = bm25s.BM25(); ret.index(bm25s.tokenize(texts, stopwords="en"), show_progress=False)
        docs, sc = ret.retrieve(bm25s.tokenize([r["query"]], stopwords="en", show_progress=False), k=n, show_progress=False)
        bm = np.zeros(n); bm[docs[0]] = sc[0]; de = D[k:k + n] @ Qe[i]; k += n
        cands = [{"pid": c["pid"], "gain": G[c["label"]], "fields": {"color": c["color"] or "", "brand": c["brand"] or "", **{t: "" for t in TYPES if t not in ("color", "brand")}},
                  "bm25": float(bm[j]), "dense": float(de[j]), "title": c["title"]} for j, c in enumerate(r["cands"])]
        out.append({"qid": f"esci_{r['qid']}", "query": r["query"], "cands": cands})
    pickle.dump(out, open(f"{DATA}/esci_cands.pkl", "wb")); print("esci", len(out))

if __name__ == "__main__":
    {"wands": wands, "esci": esci}[sys.argv[1]]()
