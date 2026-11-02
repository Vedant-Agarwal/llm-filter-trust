"""QueryNER (bltlab/queryner, CC-BY-4.0) BIO -> attribute lists in our schema. creator->brand; UoM/quantity spans->size."""
import json, os, datasets
DATA = os.environ.get("ISIR_DATA", "data")
MAP = {"color": "color", "creator": "brand", "material": "material", "shape": "shape", "UoM": "size", "quantity": "size"}
def spans(tokens, tags, names):
    out, cur, typ = [], [], None
    for t, g in zip(tokens, tags):
        g = names[g]
        if g.startswith("B-"):
            if cur: out.append((typ, " ".join(cur)))
            typ, cur = g[2:], [t]
        elif g.startswith("I-") and cur and g[2:] == typ:
            cur.append(t)
        else:
            if cur: out.append((typ, " ".join(cur)))
            cur, typ = [], None
    if cur: out.append((typ, " ".join(cur)))
    return out
def build(split):
    ds = datasets.load_dataset("bltlab/queryner", split=split, revision="f2f9173d14243496dbcf017a9dc2ded74989a45e")
    names = ds.features["ner_tags"].feature.names
    rows = []
    for i, r in enumerate(ds):
        attrs = [{"type": MAP[t], "value": v} for t, v in spans(r["tokens"], r["ner_tags"], names) if t in MAP]
        rows.append({"qid": f"qner_{split}_{i}", "query": " ".join(r["tokens"]), "gold": attrs})
    return rows
if __name__ == "__main__":
    for s in ["train", "validation", "test"]:
        rows = build(s)
        with open(f"{DATA}/queryner_{s}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(s, len(rows), sum(len(r["gold"]) > 0 for r in rows))
