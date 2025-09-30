"""
finance_rerank_core.py

Purpose
-------
Core implementation for a human-aligned retrieval & zero-shot reranking pipeline
for financial QA over long corporate reports. It mirrors the analyst workflow:
  1) Record evidence as structured "cards" (entity, metric, fiscal period, number, span).
  2) Clarify questions into "intents" that specify required evidence.
  3) Run a staged pipeline: initial screen -> global review -> tie-breaks with finance-aware criteria.
The goal is to keep token/latency predictable, expose traceable rationales, reduce temporal/numeric
mismatches, and yield stable top-k results.

Notes
-----
- This file is distilled from a Jupyter notebook provided by the author.
- Non-essential notebook elements (magics, environment setup) were removed.
- Key functions/classes retain their original logic with clearer structure and comments.
- Add your own I/O glue around `run_pipeline(...)` as needed.

Usage (CLI)
-----------
    python finance_rerank_core.py --reports ./reports --questions ./questions.json --topk 20

Dependencies
------------
Third-party imports inferred automatically below. Verify and pin versions as needed.
"""


# ===== Cell 1: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
from google.colab import drive
drive.mount('/content/drive')


# ===== Cell 2: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
import pandas as pd
import matplotlib.pyplot as plt
import json


# ===== Cell 3: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
import json

path = "/content/drive/MyDrive/MyKaggle/data/processed/cards.jsonl"

cards = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            cards.append(json.loads(line))

print("Loaded", len(cards), "cards")


# ===== Cell 4: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
cards_df = pd.DataFrame(cards)


# ===== Cell 5: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
cards_df.head()


# ===== Cell 6: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
cards_df['chunk_uid'] = cards_df['chunk_uid'].astype(int)


# ===== Cell 7: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
sample_ids = pd.read_csv("/content/drive/MyDrive/MyKaggle/data/processed/sample_ids_2000.csv")["sample_id"].tolist()


# ===== Cell 8: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
len(sample_ids)


# ===== Cell 9: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
q_map_df = pd.read_pickle("/content/drive/MyDrive/MyKaggle/data/processed/qid_map.pkl")


# ===== Cell 10: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
unique_question_df = pd.read_pickle("/content/drive/MyDrive/MyKaggle/data/processed/query_embedding_2000.pkl")


# ===== Cell 11: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
unique_question_df.head()


# ===== Cell 12: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
len(unique_question_df)


# ===== Cell 13: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
few_samples_df = pd.read_pickle("/content/drive/MyDrive/MyKaggle/data/processed/few_samples_df.pkl")


# ===== Cell 14: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
few_samples_df.head()


# ===== Cell 15: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
few_samples_df[['sample_id', 'chunk_index', 'chunk_uid']].head()


# ===== Cell 16: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
few_sample_id = few_samples_df['sample_id'].unique()


# ===== Cell 17: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
few_samples_df['sample_id'].nunique()


# ===== Cell 18: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
len(sample_ids)


# ===== Cell 19: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
from openai import OpenAI
key = "sk-Q703dTVkVheKa81jIVEZZuTiBdVrEwRt0uhGd32WnH56YuVk"
client = OpenAI(base_url="https://api2.aigcbest.top/v1", api_key=key)


# ===== Cell 20: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
questionid_to_details = (
    unique_question_df
    .set_index("uid_question")[["question", "ques_result"]]
    .to_dict("index")
)


# ===== Cell 21: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
questionid_to_details[0]


# ===== Cell 22: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
sample_to_chunks = (
    few_samples_df.groupby("sample_id")[["chunk_index", "chunk_uid"]]
    .apply(lambda x: dict(zip(x["chunk_index"], x["chunk_uid"])))
    .to_dict()
)


# ===== Cell 23: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
chunk_lookup = {
    (row.sample_id, row.chunk_uid): row.chunk_index
    for row in few_samples_df.itertuples(index=False)
}


# ===== Cell 24: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
chunk_lookup.get((27, 715))


# ===== Cell 25: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
sample_to_uidq = dict(zip(unique_question_df["sample_id"], unique_question_df["uid_question"]))


# ===== Cell 26: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
import json, re, math
import pandas as pd
from collections import Counter

def tok(s):
    return re.findall(r"[a-zA-Z0-9%$\.]{2,}", (s or "").lower())

def normalize_period_tag(pnorm: str):
    if not isinstance(pnorm, str): return None
    p = pnorm.strip()
    if re.fullmatch(r"FY\d{4}(-Q[1-4])?", p):
        return p
    m = re.fullmatch(r"(\d{4})-\d{2}-\d{2}", p)
    return m.group(1) if m else None

def period_match(q_time, pnorm: str):
    tag = normalize_period_tag(pnorm)
    if not tag:  
        return True
    fy = (q_time or {}).get("fy")
    q  = (q_time or {}).get("quarter")
    if fy and q:    return tag == f"FY{fy}-Q{q}"
    if fy and not q:return tag.startswith(f"FY{fy}")
    return True

class TinyBM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.df = Counter()
        self.avgdl = sum(len(d) for d in docs_tokens) / max(1, self.N)
        for d in docs_tokens:
            for w in set(d):
                self.df[w] += 1
        self.idf = {w: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for w, df in self.df.items()}

    def score(self, q_tokens, doc_tokens):
        if not doc_tokens: return 0.0
        score = 0.0
        dl = len(doc_tokens)
        tf = Counter(doc_tokens)
        for w in q_tokens:
            if w not in self.idf:
                continue
            f = tf.get(w, 0)
            if f == 0:
                continue
            idf = self.idf[w]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * f * (self.k1 + 1) / denom
        return score

def build_index_rows(df: pd.DataFrame):
    texts, tokens = [], []
    for _, r in df.iterrows():
        bag = " ".join([
            r["summary"] or "",
            " ".join(r["entities"]),
            " ".join(r["metrics"]),
            " ".join(r["numbers"]),
        ])
        texts.append(bag)
        tokens.append(tok(bag))
    return texts, tokens

def candidate_retrieval_for_question(query_obj: dict,
                                     allowed_chunk_ids,
                                     cards_df: pd.DataFrame,
                                     topk=50,
                                     topic_weight=1.0,
                                     entity_bonus=0.6, metric_bonus=0.4, number_bonus=0.3):
    q_topic   = (query_obj.get("topic") or "Other")
    q_ents    = set([e.lower() for e in (query_obj.get("entities") or [])])
    q_mets    = set([m.lower() for m in (query_obj.get("metrics") or [])])
    q_keys    = [w.lower() for w in (query_obj.get("keywords") or [])]
    q_time    = query_obj.get("time_filters") or {}
    q_numintent = (query_obj.get("numeric_intent") or {}).get("needs_number", False)

    allowed_set = set(str(x) for x in (allowed_chunk_ids or []))
    sub = cards_df[cards_df["chunk_uid"].astype(str).isin(allowed_set)].copy()
    if sub.empty:
        return sub.assign(score=0.0).head(0)

    sub = sub[~sub["boilerplate"]].copy()
    if sub.empty:
        return sub.assign(score=0.0).head(0)

    keep = sub[(sub["topic"] == q_topic)] if q_topic != "Other" else sub
    keep = keep[keep["period_norm"].map(lambda p: period_match(q_time, p))]
    if keep.empty: keep = sub

    texts, tokens = build_index_rows(keep)
    bm25 = TinyBM25(tokens)
    q_tok = list(set(q_keys)) or tok(query_obj.get("raw",""))
    bm_scores = [bm25.score(q_tok, dtok) for dtok in tokens]

    def overlap_boost(r):
        ents = set([e.lower() for e in r["entities"]])
        mets = set([m.lower() for m in r["metrics"]])
        boost = 0.0
        if len(q_ents & ents) > 0: boost += entity_bonus
        if len(q_mets & mets) > 0: boost += metric_bonus
        if q_numintent and len(r["numbers"]) > 0: boost += number_bonus
        if r["topic"] == q_topic: boost += 0.2 * topic_weight
        return boost

    boosts = [overlap_boost(keep.iloc[i]) for i in range(len(keep))]
    final_scores = [bm_scores[i] + boosts[i] for i in range(len(keep))]
    keep = keep.assign(score=final_scores).sort_values("score", ascending=False).head(topk)
    return keep[["chunk_uid","score","topic","section","summary","entities","metrics","numbers","period_norm"]]

def _extract_json_block(text: str) -> str:
    if not text: return ""
    t = text.strip()
    if t.startswith("{") and t.endswith("}"): return t
    l, r = t.find("{"), t.rfind("}")
    return t[l:r+1] if (l != -1 and r != -1 and r > l) else ""

def llm_rerank(question_text: str, cand_df: pd.DataFrame, client, model="gpt-4o-mini", k=10):
    if cand_df.empty: return []
    items = []
    for i, r in cand_df.head(40).iterrows():  
        items.append({
            "chunk_uid": str(r["chunk_uid"]),
            "section": r["section"],
            "topic": r["topic"],
            "summary": r["summary"],
            "entities": r["entities"],
            "metrics": r["metrics"],
            "numbers": r["numbers"],
        })
    sys = "You are a concise financial retrieval judge. Output STRICT JSON only."
    usr = f"""Question:
    {question_text}

    You are given candidate chunks as compact cards. Rank them by how well they answer the question.

    Rules:
    - Prefer exact topical and entity alignment.
    - If the question implies a numeric answer (e.g., YoY/QoQ/FY mentions), DOWN-RANK candidates with no numeric values in their 'numbers' field.
    - Break ties by specificity and evidence (less boilerplate, more concrete signals and figures).
    - Use the following financial intuition as SOFT tie-breakers (do not hard filter; only prioritize when relevance is similar):
      • Revenue / Guidance → prefer sections like MD&A, Outlook, CFO commentary, Segment.
      • Costs / Expenses / Margin → prefer Cost of sales, Gross margin, Operating expense.
      • Liquidity / Leverage / Buyback / Dividend → prefer Liquidity & Capital Resources, Capital Allocation.
      • ESG / Sustainability → prefer ESG, Sustainability, Risk.
      • Legal / Proxy / Shareholder vote → prefer Proxy, Legal.

    Output:
    Return STRICT JSON exactly as: {{"top_k": ["<chunk_uid>", ...]}} with at most {k} items.

    Candidates:
    {json.dumps(items, ensure_ascii=False)}
    """

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=200,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}]
    )
    raw = resp.choices[0].message.content
    js  = json.loads(_extract_json_block(raw))
    return [str(x) for x in js.get("top_k", [])][:k]

def get_top_chunks_for_question(question_uid: str,
                                allowed_chunk_ids,
                                unique_question_df: pd.DataFrame,
                                cards_df: pd.DataFrame,
                                k=10,
                                use_llm=True,
                                client=None,
                                model="gpt-3.5-turbo-0125"):
    row = unique_question_df.loc[unique_question_df["uid_question"] == question_uid]
    if row.empty:
        raise ValueError(f"question_uid {question_uid} not found in unique_question_df.")
    qobj = row.iloc[0]["ques_result"]
    qtxt = row.iloc[0].get("question", qobj.get("raw",""))

    cands = candidate_retrieval_for_question(qobj, allowed_chunk_ids, cards_df, topk=max(50, k))

    if not use_llm:
        return cands["chunk_uid"].astype(str).head(k).tolist()

    if client is None:
        raise ValueError("use_llm=True but client is None.")
    top_ids = llm_rerank(qtxt, cands, client=client, model=model, k=k)

    if len(top_ids) < k:
        fallback = [x for x in cands["chunk_uid"].astype(str).tolist() if x not in top_ids]
        top_ids += fallback[:(k - len(top_ids))]
    return top_ids


# ===== Cell 27: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
MODEL_RERANK = "gpt-3.5-turbo-0125"

import json, re
from typing import List

def _extract_json_block(text: str) -> str:
    if not text: return ""
    t = text.strip()
    if t.startswith("{") and t.endswith("}"): return t
    l, r = t.find("{"), t.rfind("}")
    return t[l:r+1] if (l != -1 and r != -1 and r > l) else ""

def _get_nested(r, path, default=None):
    
    cur = r
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            # r may be a pandas.Series; try to access flattened columns directly
            flat = "_".join(path)  # e.g., card_topic
            if flat in r: return r[flat]
            if key in r:  cur = r[key]
            else:         return default
    return cur

def _row_to_item_robust(r):
    
    def g(paths, default=None):
        # paths is a list of alternative paths; return the first that matches
        for p in paths:
            val = _get_nested(r, p, None)
            if val is not None:
                return val
        return default

    return {
        "chunk_uid": str(g([["chunk_uid"]])),
        "topic":      g([["card","topic"],["topic"]], "Other"),
        "section":    g([["card","section"],["section"]], "Other"),
        "summary":    (g([["card","summary"],["summary"]], "") or "")[:220],
        "entities":   g([["card","entities"],["entities"]], [])[:4] or [],
        "metrics":    g([["card","metrics"],["metrics"]], [])[:4] or [],
        "numbers":    g([["card","numbers"],["numbers"]], [])[:4] or [],
        "period":     g([["card","period_normalized"],["period_norm"],["card","period"],["period"]], "N/A"),
        "boilerplate_flag": bool(g([["domain","boilerplate_flag"],["boilerplate_flag"]], False)),
    }

# ========== LLM sub-steps ==========
def _llm_pick_top(question_text: str, items: List[dict], top_m: int) -> List[str]:
    
    sys = (
    "You are a concise financial retrieval judge. Output STRICT JSON only. "
    "Judge relevance for answering the question."
    )

    usr = f"""Question:
    {question_text}

    You will receive candidate chunks as compact 'cards' with fields:
    [chunk_uid, topic, section, period, entities, metrics, numbers, summary, boilerplate_flag]

    Instructions:
    - Rank the chunks by how well they answer the question.
    - Respect temporal intent in the question if any (e.g., FY/quarter/year mentions).
    - Prefer exact topical/entity/metric alignment over loose mentions.
    - If the question implies a numeric answer (e.g., YoY/QoQ/FY mentions), DOWN-RANK candidates with no numeric values in their 'numbers' field.
    - Deprioritize boilerplate or generic introductions unless they directly answer the question.
    - Apply the following financial intuition as SOFT tie-breakers (do not hard filter; only prioritize when relevance is similar):
      • Revenue / Guidance → prefer MD&A, Outlook, CFO commentary, Segment.
      • Costs / Expenses / Margin → prefer Cost of sales, Gross margin, Operating expense.
      • Liquidity / Leverage / Buyback / Dividend → prefer Liquidity & Capital Resources, Capital Allocation.
      • ESG / Sustainability → prefer ESG, Sustainability, Risk.
      • Legal / Proxy / Shareholder vote → prefer Proxy, Legal.

    Return STRICT JSON exactly as: {{"top": ["<chunk_uid>", ...]}} with at most {top_m} items.

    Candidates:
    {json.dumps(items, ensure_ascii=False)}
    """

    resp = client.chat.completions.create(
        model=MODEL_RERANK,
        temperature=0,
        max_tokens=1000,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}]
    )
    raw = resp.choices[0].message.content
    js  = json.loads(_extract_json_block(raw) or "{}")
    return [str(x) for x in js.get("top", [])][:top_m]

def _llm_final_rank(question_text: str, finalists: List[dict], k: int) -> List[str]:
   
    sys = "You are a concise financial retrieval judge. Output STRICT JSON only."
    usr = f"""Question:
{question_text}

FINALIST chunks (cards):
{json.dumps(finalists, ensure_ascii=False)}

Rank by: ability to answer the question, temporal fit, specificity, numeric evidence (when implied).
Return STRICT JSON: {{"top_k": ["<chunk_uid>", ...]}} (max {k} items).
"""
    resp = client.chat.completions.create(
        model=MODEL_RERANK,
        temperature=0,
        max_tokens=1000,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}]
    )
    raw = resp.choices[0].message.content
    js  = json.loads(_extract_json_block(raw) or "{}")
    return [str(x) for x in js.get("top_k", [])][:k]


# ===== Cell 28: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
import random
from typing import List, Tuple

def _safe_int(u):
    try:
        return int(str(u))
    except Exception:
        m = re.search(r"\d+", str(u))
        return int(m.group(0)) if m else 10**9

def _schedule_pairs(uids: List[str], max_matches_per_uid: int = 6, seed: int = 42) -> List[Tuple[str, str]]:
   
    rng = random.Random(seed)
    u = uids[:]
    rng.shuffle(u)
    pairs = []
    played = {x: 0 for x in u}
    for i in range(len(u)):
        a, b = u[i], u[(i+1) % len(u)]
        if a == b:
            continue
        if played[a] < max_matches_per_uid and played[b] < max_matches_per_uid:
            pairs.append((a, b))
            played[a] += 1; played[b] += 1
    attempts = 0
    while attempts < 5*len(u):
        a, b = rng.sample(u, 2)
        if a == b:
            continue
        if played[a] >= max_matches_per_uid or played[b] >= max_matches_per_uid:
            attempts += 1;
            continue
        pairs.append((a, b))
        played[a] += 1; played[b] += 1
        attempts += 1
    seen = set()
    uniq = []
    for a, b in pairs:
        k = tuple(sorted([a, b]))
        if k in seen:
            continue
        seen.add(k); uniq.append((a, b))
    return uniq

def _llm_pairwise_winner(question_text: str, A: dict, B: dict) -> str:
    
    sys = "You are a concise financial retrieval judge. Output STRICT JSON only."
    usr = f"""Question:
{question_text}

Compare which single chunk better answers the question. Consider relevance, temporal fit, numeric evidence (if implied), and specificity.
Return STRICT JSON exactly as: {{"winner": "A"}} or {{"winner": "B"}}.

A = {json.dumps(A, ensure_ascii=False)}
B = {json.dumps(B, ensure_ascii=False)}
"""
    resp = client.chat.completions.create(
        model=MODEL_RERANK,
        temperature=0,
        max_tokens=50,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}]
    )
    raw = resp.choices[0].message.content
    js  = json.loads(_extract_json_block(raw) or "{}")
    w = (js.get("winner") or "").strip().upper()
    return 'A' if w == 'A' else 'B'

def _pairwise_judge_rank(question_text: str, finalists: List[dict], max_matches_per_uid: int = 6) -> List[str]:
    
    uids = [it["chunk_uid"] for it in finalists]
    rating = {u: 1500.0 for u in uids}
    pairs = _schedule_pairs(uids, max_matches_per_uid=max_matches_per_uid)
    K = 24
    uid2item = {it["chunk_uid"]: it for it in finalists}
    for ua, ub in pairs:
        A, B = uid2item[ua], uid2item[ub]
        winner = _llm_pairwise_winner(question_text, A, B)
        Ra, Rb = rating[ua], rating[ub]
        Ea = 1.0 / (1 + 10 ** ((Rb - Ra) / 400))
        Eb = 1.0 - Ea
        Sa, Sb = (1, 0) if winner == 'A' else (0, 1)
        rating[ua] = Ra + K * (Sa - Ea)
        rating[ub] = Rb + K * (Sb - Eb)
    return sorted(uids, key=lambda u: rating[u], reverse=True)

def _rrf_fuse(rank_lists: List[List[str]], k: int = 60) -> Tuple[List[str], dict]:
    
    score = {}
    for r in rank_lists:
        for idx, u in enumerate(r):
            score[u] = score.get(u, 0.0) + 1.0 / (k + idx + 1)
    fused = sorted(score.keys(), key=lambda u: score[u], reverse=True)
    return fused, score

def _entity_match(card: dict, qobj: dict) -> float:
    qset = set([e.lower() for e in (qobj.get("entities") or [])])
    cset = set([e.lower() for e in (card.get("entities") or [])])
    return 1.0 if (qset and (qset & cset)) else (0.5 if not qset else 0.0)

def _metric_match(card: dict, qobj: dict) -> float:
    qset = set([m.lower() for m in (qobj.get("metrics") or [])])
    cset = set([m.lower() for m in (card.get("metrics") or [])])
    return 1.0 if (qset and (qset & cset)) else (0.5 if not qset else 0.0)

def _period_compatible(card: dict, qobj: dict) -> float:
    q_time = qobj.get("time_filters") or {}
    p = card.get("period") or card.get("period_norm") or "N/A"
    return 1.0 if period_match(q_time, p) else 0.0

def _align_score(card: dict, qobj: dict) -> float:
    return _entity_match(card, qobj) + _metric_match(card, qobj) + _period_compatible(card, qobj)

def _diversify(uids: List[str], uid2item: dict, section_quota: int = 3, neighbor_gap: int = 3) -> List[str]:
    kept, sec_count = [], {}
    for u in uids:
        sec = (uid2item[u].get("section") or "NA")
        if sec_count.get(sec, 0) >= section_quota:
            continue
        uo = _safe_int(u)
        too_close = any(abs(uo - _safe_int(v)) < neighbor_gap and (uid2item[v].get("section") or "NA") == sec for v in kept)
        if too_close:
            continue
        kept.append(u)
        sec_count[sec] = sec_count.get(sec, 0) + 1
    return kept

def get_top_chunks_for_sample_llm(sample_id, k: int = 10, batch_size: int = 30, per_batch: int = 6,
                                  bagging_rounds: int = 1,  # no extra cost by default; set 2-3 to enable resampling
                                  section_quota: int = 3, neighbor_gap: int = 3):
    
    if sample_id not in sample_to_uidq:
        raise ValueError(f"sample_id {sample_id} not in sample_to_uidq")

    print(f"Starting sample {sample_id}!")
    question_uid = sample_to_uidq[sample_id]
    print(f"question_uid: {question_uid}")

    temp_question_detail = questionid_to_details[question_uid]
    qobj = temp_question_detail["ques_result"]
    qtxt = temp_question_detail["question"]

    # In-document chunk_uids
    idx_list = sample_to_chunks[sample_id].keys()
    allowed_chunk_ids = [sample_to_chunks[sample_id][idx] for idx in idx_list]
    print(f"chunk_id:{idx_list}")
    print(f"length_chunk:{len(idx_list)}")
    print(f"allowed_chunk_ids:{allowed_chunk_ids}")
    print(f"length_allowed_chunk_ids:{len(allowed_chunk_ids)}")

    if not allowed_chunk_ids:
        raise ValueError(f"No chunk_uids found for sample_id {sample_id}")

    # Take cards for these chunks
    sub = cards_df[cards_df["chunk_uid"].isin(allowed_chunk_ids)].copy()
    items_all = [_row_to_item_robust(r) for _, r in sub.iterrows()]
    print(f"len of sub is {len(sub)}")

    # ========== 4) Group elimination (optional bagging) ==========
    rng = random.Random(42)
    finalists = []
    if len(items_all) <= k:
        finalists = items_all
    else:
        rounds = max(1, int(bagging_rounds))
        for r in range(rounds):
            items = items_all[:]
            if rounds > 1:
                rng.shuffle(items)  # shuffle for resampling
            for i in range(0, len(items), batch_size):
                group = items[i:i+batch_size]
                try:
                    picked_ids = _llm_pick_top(qtxt, group, top_m=min(per_batch, len(group)))
                except Exception:
                    picked_ids = []
                uid2item_local = {it["chunk_uid"]: it for it in group}
                finalists.extend([uid2item_local[u] for u in picked_ids if u in uid2item_local])
                print(f"[Round {r}] Group {i}: finalists are {picked_ids}")

        # If finalists are insufficient, pad with more (will be re-filtered in finals)
        if len(finalists) < k:
            seen = set(it["chunk_uid"] for it in finalists)
            for it in items_all:
                if it["chunk_uid"] not in seen:
                    finalists.append(it)
                if len(finalists) >= max(k, 2*per_batch):
                    break

    # Deduplicate and keep the most recent info
    uid2item = {}
    for it in finalists:
        uid2item[it["chunk_uid"]] = it
    finalists = list(uid2item.values())

    print(f"{len(finalists)} finalists after prelims.")

    # ========== 5) Finals: listwise rerank + Pairwise → RRF ==========
    if len(finalists) == 0:
        return []

    finalists_subset = finalists[: max(k, 5*per_batch)]  # control context size
    # Listwise judge: return as many ranks as possible for rank_L
    rank_L = _llm_final_rank(qtxt, finalists_subset, k=len(finalists_subset))
    # Pairwise judge: only duel top-20 to control cost
    top_for_pairwise = rank_L[: min(20, len(rank_L))] if len(rank_L) > 0 else [it["chunk_uid"] for it in finalists_subset]
    pairwise_input = [uid2item[u] for u in top_for_pairwise]
    rank_P = _pairwise_judge_rank(qtxt, pairwise_input, max_matches_per_uid=6)

    # Append remaining uids not in pairwise to the tail of rank_P to ensure domain coverage
    tail = [u for u in [it["chunk_uid"] for it in finalists_subset] if u not in rank_P]
    rank_P = rank_P + tail

    fused_uids, rrf_scores = _rrf_fuse([rank_L, rank_P], k=60)

    # ========== 6) Post-hoc: alignment scoring + diversity constraints ==========
    alpha = 0.5  # alignment score weight
    scored = []
    for u in fused_uids:
        card = uid2item.get(u)
        if not card:
            continue
        a = _align_score(card, qobj)  # 0..3
        s = rrf_scores.get(u, 0.0) + alpha * a
        scored.append((u, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = [u for u, _ in scored]

    ranked = _diversify(ranked, uid2item, section_quota=section_quota, neighbor_gap=neighbor_gap)

    final_ids = ranked[:k]

    # ========== 7) Fallback padding ==========
    if len(final_ids) < k:
        exist = set(final_ids)
        for it in finalists_subset:
            if it["chunk_uid"] not in exist:
                final_ids.append(it["chunk_uid"])
            if len(final_ids) >= k:
                break

    temp_res = final_ids[:k]
    print(f"temp_res{temp_res}")
    chunkid_list = [chunk_lookup[(sample_id, int(uid))] for uid in temp_res]
    print(chunkid_list)
    return chunkid_list


# ===== Cell 29: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
some_sample_id = few_sample_id[1]
print(f"Selected sample {some_sample_id}")
top10 = get_top_chunks_for_sample_llm(
    sample_id=some_sample_id,
    k=10,
    batch_size=30,  # 30 cards per batch for LLM
    per_batch=6     # select 6 finalists per batch
)
print("Top-10 chunks:", top10)


# ===== Cell 30: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
from tqdm.auto import tqdm
import time, json, csv, os
from datetime import datetime

# ======== Configure output paths ========
OUT_DIR = "/content/drive/MyDrive/MyKaggle/data/processed/batch_run_pairwise_optimization"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH   = f"{OUT_DIR}/top10_results.csv"
JSONL_PATH = f"{OUT_DIR}/top10_results.jsonl"
FAIL_LOG   = f"{OUT_DIR}/failures.log"
CKPT_PATH  = f"{OUT_DIR}/checkpoint.json"  # record completed sample_ids; supports resuming

# ======== Optional: restore from checkpoint ========
done = set()
if os.path.exists(CKPT_PATH):
    try:
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            done = set(json.load(f))
    except Exception:
        done = set()

# ======== CSV header (write if not exists) ========
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "question_uid", "top10_uids", "batch_size", "per_batch", "ts"])

def append_csv_row(row):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)

def append_jsonl(obj):
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_fail(sample_id, err):
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}Z] sample_id={sample_id} ERROR: {err}\n")

def save_ckpt(done_set):
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(list(done_set)), f, ensure_ascii=False)

# ======== Batch main loop ========
def batch_run(few_sample_id, k=10, batch_size=30, per_batch=6, max_retries=0, sleep_sec=1.5):
    total = len(few_sample_id)
    pbar = tqdm(few_sample_id, desc="Running samples", unit="sample")

    for sample_id in pbar:
        if sample_id in done:
            pbar.set_postfix_str(f"skip {sample_id}")
            continue

        # question_uid (for logging to disk)
        try:
            q_uid = sample_to_uidq[sample_id]
        except KeyError:
            append_fail(sample_id, "question_uid not found in sample_to_uidq")
            continue

        # Retry mechanism (more robust to network/API errors)
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                top10 = get_top_chunks_for_sample_llm(
                    sample_id=sample_id,
                    k=k,
                    batch_size=batch_size,
                    per_batch=per_batch
                )
                # On success: write CSV/JSONL
                append_csv_row([sample_id, q_uid, "|".join(map(str, top10)), batch_size, per_batch, datetime.utcnow().isoformat()+"Z"])
                append_jsonl({
                    "sample_id": sample_id,
                    "question_uid": q_uid,
                    "top_k": top10,
                    "k": k,
                    "batch_size": batch_size,
                    "per_batch": per_batch,
                    "ts": datetime.utcnow().isoformat()+"Z"
                })
                done.add(sample_id)
                save_ckpt(done)
                pbar.set_postfix_str(f"ok {sample_id}")
                break
            except Exception as e:
                last_err = e
                pbar.set_postfix_str(f"retry {attempt+1}/{max_retries}")
                time.sleep(sleep_sec)

        # If retries still fail, log the error
        if last_err is not None and sample_id not in done:
            append_fail(sample_id, repr(last_err))

    # Final report
    print(f"\n✅ Finished. CSV: {CSV_PATH}\nJSONL: {JSONL_PATH}\nFailures: {FAIL_LOG}\nCheckpoint: {CKPT_PATH}")


# ===== Cell 31: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
batch_run(few_sample_id, k=10, batch_size=30, per_batch=6)


# ===== Cell 32: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
csv_path = "/content/drive/MyDrive/MyKaggle/data/processed/batch_run_pairwise_optimization/top10_results.csv"
df = pd.read_csv(csv_path)
print("Total samples:", len(df))
print("Head:")
print(df.head())


# ===== Cell 33: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df


# ===== Cell 34: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df['sample_id'].nunique()


# ===== Cell 35: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
temp_id = df['sample_id']


# ===== Cell 36: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
len(few_sample_id), len(set(few_sample_id))


# ===== Cell 37: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
jsonl_path = "/content/drive/MyDrive/MyKaggle/data/processed/batch_run_pairwise_optimization/top10_results.jsonl"
with open(jsonl_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(json.loads(line))


# ===== Cell 38: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
fail_log = "/content/drive/MyDrive/MyKaggle/data/processed/batch_run_pairwise_optimization/failures.log"
with open(fail_log, "r", encoding="utf-8") as f:
    fails = f.readlines()
print("Failure count:", len(fails))
if fails:
    print("Sample errors:", fails[:3])


# ===== Cell 39: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df = df.drop_duplicates(subset=["sample_id", "question_uid"], keep="last")
df


# ===== Cell 40: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df["top10_list"] = df["top10_uids"].astype(str).str.split("|")


# ===== Cell 41: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df


# ===== Cell 42: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
def extract_labels(sample_id):
    data = q_map_df[sample_id]
    golden_list = data["golden_label_list"]
    label1 = [cid for cid, lbl in golden_list if lbl == 1]
    label2 = [cid for cid, lbl in golden_list if lbl == 2]
    return label1, label2

df["label1_chunks"], df["label2_chunks"] = zip(
    *df["sample_id"].apply(extract_labels)
)


# ===== Cell 43: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df


# ===== Cell 44: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
# Ensure the three columns are list[int]
df["top10_list"] = df["top10_list"].apply(lambda x: list(map(int, x)))
df["label1_chunks"] = df["label1_chunks"].apply(lambda x: list(map(int, x)))
df["label2_chunks"] = df["label2_chunks"].apply(lambda x: list(map(int, x)))


# ===== Cell 45: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
def compute_stats(row):
    top10 = set(row["top10_list"])
    l1 = set(row["label1_chunks"])
    l2 = set(row["label2_chunks"])

    # 1 & 2. Hit counts
    l1_hits = len(l1 & top10)
    l2_hits = len(l2 & top10)

    # 3 & 4. Total lengths
    l1_len = len(l1)
    l2_len = len(l2)

    # 5. Hit ratio
    total_gt = l1_len + l2_len
    total_hits = l1_hits + l2_hits
    hit_ratio = total_hits / total_gt if total_gt > 0 else 0

    return pd.Series([l1_hits, l2_hits, l1_len, l2_len, hit_ratio])

df[["l1_hits", "l2_hits", "l1_len", "l2_len", "hit_ratio"]] = df.apply(compute_stats, axis=1)


# ===== Cell 46: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df


# ===== Cell 47: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df['hit_ratio'].describe()


# ===== Cell 48: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
import numpy as np

def dcg(rels, k):
    return sum(rel / np.log2(idx+2) for idx, rel in enumerate(rels[:k]))

def ndcg_at_k(topk, relevant, k=5):
    rels = [1 if x in relevant else 0 for x in topk[:k]]
    dcg_val = dcg(rels, k)
    ideal_rels = sorted(rels, reverse=True)
    idcg_val = dcg(ideal_rels, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0

def ap_at_k(topk, relevant, k=5):
    rels = [1 if x in relevant else 0 for x in topk[:k]]
    if sum(rels) == 0:
        return 0
    precisions = [sum(rels[:i+1])/(i+1) for i in range(k) if rels[i] == 1]
    return np.mean(precisions)

def rr_at_k(topk, relevant, k=5):
    for i, x in enumerate(topk[:k]):
        if x in relevant:
            return 1/(i+1)
    return 0

# Apply to df
df["ndcg@5"] = df.apply(lambda r: ndcg_at_k(r["top10_list"], set(r["label1_chunks"]+r["label2_chunks"]), 5), axis=1)
df["map@5"]  = df.apply(lambda r: ap_at_k(r["top10_list"], set(r["label1_chunks"]+r["label2_chunks"]), 5), axis=1)
df["mrr@5"]  = df.apply(lambda r: rr_at_k(r["top10_list"], set(r["label1_chunks"]+r["label2_chunks"]), 5), axis=1)

# Take average
metrics = df[["ndcg@5","map@5","mrr@5"]].mean()
print(metrics)


# ===== Cell 49: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
import numpy as np

def dcg(rels, k):
    return sum(rel / np.log2(idx+2) for idx, rel in enumerate(rels[:k]))

def ndcg_at_k(topk, relevant, k=5):
    rels = [1 if x in relevant else 0 for x in topk[:k]]
    dcg_val = dcg(rels, k)
    ideal_rels = sorted([1]*min(len(relevant), k) + [0]*(k - min(len(relevant), k)), reverse=True)
    idcg_val = dcg(ideal_rels, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0

def map_at_k(topk, relevant, k=5):
    rels = [1 if x in relevant else 0 for x in topk[:k]]
    precisions, hit_count = [], 0
    for i, r in enumerate(rels, 1):
        if r == 1:
            hit_count += 1
            precisions.append(hit_count / i)
    return np.mean(precisions) if precisions else 0

def mrr_at_k(topk, relevant, k=5):
    for i, x in enumerate(topk[:k], 1):
        if x in relevant:
            return 1 / i
    return 0

# === Apply to the entire df ===
df["ndcg@5"] = df.apply(lambda r: ndcg_at_k(r["top10_list"], set(r["label1_chunks"] + r["label2_chunks"]), 5), axis=1)
df["map@5"]  = df.apply(lambda r: map_at_k(r["top10_list"], set(r["label1_chunks"] + r["label2_chunks"]), 5), axis=1)
df["mrr@5"]  = df.apply(lambda r: mrr_at_k(r["top10_list"], set(r["label1_chunks"] + r["label2_chunks"]), 5), axis=1)


# ===== Cell 50: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df


# ===== Cell 51: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df[["ndcg@5", "map@5", "mrr@5"]].describe()


# ===== Cell 52: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
df.to_pickle("/content/drive/MyDrive/MyKaggle/data/processed/results_2000_LLM_pairwise.pkl")


# ===== Cell 53: Converted from notebook =====
# Context: Core logic or helpers preserved; adjust names as needed.
import math
import numpy as np
import pandas as pd

# ---- 1) Optional: ensure all three columns are list[int] (noop if already) ----
for col in ["top10_list", "label1_chunks", "label2_chunks"]:
    df[col] = df[col].apply(lambda x: list(map(int, x)))

# ---- 2) Merge a row's label1/label2 into {chunk_uid: relevance(0/1/2)} ----
def make_relevance_dict(row):
    rel = {cid: 1 for cid in row["label1_chunks"]}         # partially relevant -> 1
    for cid in row["label2_chunks"]:                       # strongly relevant -> 2 (overrides 1)
        rel[cid] = 2
    return rel

# ---- 3) Metric implementations (aligned with the paper) ----
def dcg_at_k(rels, k):
    # Graded relevance DCG: commonly rel / log2(rank+1)
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

def ndcg_at_k(topk, relevance_dict, k=5):
    # graded relevance 0/1/2
    rels = [relevance_dict.get(x, 0) for x in topk[:k]]
    dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(relevance_dict.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0

def map_at_k(topk, relevance_dict, k=5):
    # Binarize: >0 is considered relevant
    rels = [1 if relevance_dict.get(x, 0) > 0 else 0 for x in topk[:k]]
    precisions, hit = [], 0
    for i, r in enumerate(rels, 1):
        if r == 1:
            hit += 1
            precisions.append(hit / i)
    return float(np.mean(precisions)) if precisions else 0.0

def mrr_at_k(topk, relevance_dict, k=5):
    # Binarize: >0 is considered relevant
    for i, x in enumerate(topk[:k], 1):
        if relevance_dict.get(x, 0) > 0:
            return 1.0 / i
    return 0.0

# ---- 4) Batch compute and add three new columns ----
K = 5
def compute_row_metrics(r):
    rel_dict = make_relevance_dict(r)
    return pd.Series({
        "ndcg@5": ndcg_at_k(r["top10_list"], rel_dict, K),
        "map@5":  map_at_k(r["top10_list"], rel_dict, K),
        "mrr@5":  mrr_at_k(r["top10_list"], rel_dict, K),
    })

df[["ndcg@5", "map@5", "mrr@5"]] = df.apply(compute_row_metrics, axis=1)

# ---- 5) Optional: overall means (aligned with Table 3 summary) ----
overall = df[["ndcg@5", "map@5", "mrr@5"]].mean()
overall


# ------------------------------
# Minimal CLI wrapper (optional)
# ------------------------------
def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Financial QA retrieval & reranking pipeline")
    p.add_argument("--reports", type=str, required=False, help="Path to directory of corporate reports (txt/jsonl).")
    p.add_argument("--questions", type=str, required=False, help="Path to questions file (json/jsonl).")
    p.add_argument("--topk", type=int, default=20, help="Top-k candidates after reranking.")
    p.add_argument("--device", type=str, default=None, help="Device override for embedding/LLM if used.")
    return p

def _maybe_main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    # NOTE: Replace the following with your project-specific entrypoint if present in the code below.
    if 'run_pipeline' in globals():
        print("[info] Running pipeline...")
        out = run_pipeline(
            reports_path=args.reports,
            questions_path=args.questions,
            topk=args.topk,
            device=args.device,
        )
        # Print or save results as needed
        try:
            import json
            print(json.dumps(out, ensure_ascii=False, indent=2))
        except Exception:
            print(out)
    else:
        print("[warn] No `run_pipeline` function found. Import this module and call your own entrypoint.")

if __name__ == "__main__":
    _maybe_main()