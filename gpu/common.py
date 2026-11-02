import json
TYPES = ["color", "brand", "material", "shape", "style", "size", "pattern", "other"]
SCHEMA = {"type": "object", "properties": {"attributes": {"type": "array", "maxItems": 6, "items": {
    "type": "object", "properties": {"type": {"type": "string", "enum": TYPES}, "value": {"type": "string", "maxLength": 40}},
    "required": ["type", "value"], "additionalProperties": False}}}, "required": ["attributes"], "additionalProperties": False}
SYSTEM = ("You extract product attributes that a shopper explicitly states in an e-commerce search query. "
          "Types: color (incl. shades like navy, charcoal), brand (manufacturer, designer, store brand or collection name), "
          "material (wood, leather, cotton...), shape (round, rectangular...), style (modern, farmhouse, mid-century...), "
          "size (dimensions, bed/rug sizes, capacities with units), pattern (striped, floral...), other (any other explicit "
          "constraint). Do NOT output the product type itself (e.g. 'sofa', 'shoes'). Copy values verbatim from the query. "
          "Return JSON {\"attributes\": [...]}, possibly empty.")
def chat(tok, q):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Query: {q}"}],
                                   tokenize=False, add_generation_prompt=True)
def sampling(SP, n, temp, max_tokens=160):
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SP(n=n, temperature=temp, max_tokens=max_tokens, structured_outputs=StructuredOutputsParams(json=SCHEMA), seed=0)
    except Exception:
        from vllm.sampling_params import GuidedDecodingParams
        return SP(n=n, temperature=temp, max_tokens=max_tokens, guided_decoding=GuidedDecodingParams(json=SCHEMA), seed=0)
def parse(text):
    try:
        return [{"type": a["type"], "value": a["value"].strip()} for a in json.loads(text)["attributes"] if a.get("value", "").strip()]
    except Exception:
        return None
def load_queries(paths):
    out = []
    for p in paths:
        out += [json.loads(l) for l in open(p)]
    return out
