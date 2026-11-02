"""Table II and the worked examples in the text: confirm each row in the candidates and the LoRA extraction, and print it."""
import re
from lib import *
E = {d: load_cands(d) for d in RETAILERS}; L = load_extract("qwen3b_lora")
ROWS = [("wands", r"black freestanding tub\s+with faucet", "color", "black", r"^70 '' x 32 '' freestanding soaking fiberglass bathtub", ""),
        ("esci", r"white desk", "color", "white", r"Home Office 48-Inch Computer Desk, White/Oak", "Oak"),
        ("esci", r"master spa inner filters for hot tubs", "brand", "master spa", r"Filter for Compatible with Pleatco Spa", "Ketofa"),
        ("wands", r"7 draw white dresser", "color", "white", r"7 drawer dresser", "black"),
        ("wands", r"black freestanding tub\s+with faucet", "shape", "with faucet", r"^70 '' x 32 '' freestanding soaking fiberglass bathtub", "rectangular")]
out = []
for d, qp, t, v, tp, fv in ROWS:
    q = next(x for x in E[d] if re.fullmatch(qp, x["query"])); top = max(c["gain"] for c in q["cands"])
    extracted = any(a["type"] == t and a["value"].lower() == v for a in L[q["qid"]]["greedy"])
    hits = [c for c in q["cands"] if c["gain"] == top and re.search(tp, str(c["title"])) and c["fields"].get(t, "") == fv]
    removed = bool(hits) and not lex(v, hits[0]["fields"].get(t, ""))
    ok = extracted and removed
    out.append({"query": q["query"], "attribute": f"{t}={v}", "title": hits[0]["title"] if hits else None, "field": fv, "extracted": extracted, "top_relevance_removed": removed, "ok": ok})
    print(("OK  " if ok else "FAIL"), out[-1])
# text: 'wd' stored as a Japanese brand string, relevant product judged Exact (Section V); spa filter does not name the queried brand
wd = [(q["query"], c["fields"]["brand"]) for q in E["esci"] if re.search(r"\bwd\b", q["query"]) for c in q["cands"]
      if c["gain"] == 1 and c["fields"]["brand"] and not c["fields"]["brand"].isascii() and any(a["type"] == "brand" and a["value"].lower() == "wd" for a in (L[q["qid"]]["greedy"] or []))]
ket = next(c for q in E["esci"] if q["query"] == "master spa inner filters for hot tubs" for c in q["cands"] if c["fields"]["brand"] == "Ketofa")
print("wd example:", wd[:1]); print("spa filter title names 'master':", "master" in ket["title"].lower())
save("table2", {"rows": out, "wd_japanese_brand": bool(wd), "spa_title_names_brand": "master" in ket["title"].lower()})
