import json, os, pickle, re, random
DATA = os.environ.get("ISIR_DATA", "data")

SYN = {"grey": "gray"}
toks = lambda s: [SYN.get(t, t) for t in re.findall(r"[a-z0-9]+", str(s).lower())]
def lex(v, f):
    v, f = toks(v), set(toks(f)); return bool(v) and bool(f) and all(t in f or t.rstrip("s") in f or t + "s" in f for t in v)
EX = {json.loads(l)["qid"]: json.loads(l) for l in open("extractions/extract_qwen3b_lora.jsonl")}
rng = random.Random(1)
for d, types in (("wands", ["color", "style", "material", "shape"]), ("esci", ["color", "brand", "size"])):
    Q = pickle.load(open(f"{DATA}/{d}_cands.pkl", "rb")); rng.shuffle(Q); shown = {}
    for q in Q:
        r = EX.get(q["qid"])
        if not r or not r.get("greedy"): continue
        g = max(c["gain"] for c in q["cands"])
        for a in r["greedy"]:
            t = a["type"]
            if t not in types or shown.get(t, 0) >= 3: continue
            for c in q["cands"]:
                if c["gain"] == g and g > 0 and not lex(a["value"], c["fields"].get(t, "")):
                    f = c["fields"].get(t, "") or "(empty)"
                    print(f"{d:5s} | query: {q['query']!r:42s} | {t}={a['value']!r:14s} | relevant: {str(c['title'])[:52]!r} | field: {f[:45]!r}")
                    shown[t] = shown.get(t, 0) + 1; break
