"""
Query module extracted from notebook "4b02e5ab-2109-41f6-afbe-b441f99e822b.ipynb".

This module implements core query logic for financial QA over long corporate reports,
aligned with the paper's retrieval paradigm:
- Record evidence as structured cards (entities, metrics, periods, numbers, verbatim spans).
- Clarify questions into intents specifying required evidence.
- Use an analyst-style workflow (screen → global review → tie-breaks) to produce
  zero-shot reranking suitable for finance QA.

Auto-converted from a Jupyter notebook; exploratory/visualization/debug cells were omitted.
Light editing: IPython magics removed, comments normalized, and small helper stubs kept.

NOTE: If any top-level executable snippets existed, they've been wrapped under
      `if __name__ == "__main__":` to avoid side effects on import.
"""
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa
# mypy: ignore-errors
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

questions_df = pd.DataFrame(questions, columns=["sample_id", "question"])

import re
from typing import List, Dict

# --- Enums consistent with cards ---
TOPIC_ENUM = {"Risk","ESG","Market","Revenue","Profitability","Liquidity",
              "Costs/Expenses","Guidance/Outlook","AccountingPolicy","Legal",
              "MD&A","FinancialStatements","Other"}

# --- Vocabulary (extend based on corpus as needed) ---
METRIC_VOCAB = {
    "revenue","sales","top line","eps","earnings","margin","gross margin","operating margin",
    "cost","costs","expenses","opex","sg&a","r&d","cogs","capex","cash flow","fcf",
    "working capital","inventory","backlog","bookings","units","shipment","asp",
    "debt","interest","leverage","liquidity","guidance","outlook","churn","retention",
    "headcount","hiring","layoffs","labor","wage","salary",
    "esg","sustainability","carbon","emission","risk","litigation","legal"
}

TOPIC_HOOKS = {
    "ESG": {"esg","sustainab","carbon","emission","climate"},
    "Market": {"market","demand","customer","competition","competitor","share","pricing"},
    "Revenue": {"revenue","sales","top line","booking","backlog"},
    "Profitability": {"margin","profit","eps","gross margin","operating margin"},
    "Liquidity": {"cash","cash flow","liquidity","debt","interest","leverage","fcf"},
    "Costs/Expenses": {"cost","costs","expense","opex","sg&a","r&d","labor","wage","salary","cogs"},
    "Guidance/Outlook": {"guidance","outlook","expect","forecast"},
    "AccountingPolicy": {"accounting","policy","recognition","impairment"},
    "Legal": {"litigation","lawsuit","legal","regulatory","compliance"},
    "MD&A": {"md&a","management discussion","management’s discussion","management's discussion"},
    "FinancialStatements": {"balance sheet","income statement","cash flow statement","10-k","10-q","item 8"},
}

RELATION_LEX = {
    "increase": {"increase","rise","grow","higher","up","expand","improve"},
    "decrease": {"decrease","decline","fall","drop","lower","down","contract","deteriorate"},
    "influence": {"influence","impact","affect","drive","lead to","cause","sensitive to","elasticity"},
    "compare": {"yoy","qoq","versus","compared","vs","year over year","quarter over quarter"},
    "explain": {"why","reason","driver","attribution","explain","contribute"},
}
FY_RE  = re.compile(r"(?i)\bFY\s?(\d{4})\b|Fiscal Year\s*(\d{4})")
Q_RE   = re.compile(r"(?i)\bQ([1-4])\b")
YOY_RE = re.compile(r"(?i)\b(YoY|year over year)\b")
QOQ_RE = re.compile(r"(?i)\b(QoQ|quarter over quarter)\b")
DATE_RE= re.compile(r"(?i)\b(\w+\s+\d{1,2},\s*\d{4}|\d{4})\b")

def lower(s:str)->str: return (s or "").lower()

def pick_topic(q:str)->str:
    ql = lower(q)
    for topic, hooks in TOPIC_HOOKS.items():
        if any(h in ql for h in hooks):
            return topic
    # fallback by metric cue
    if any(k in ql for k in ["cost","expense","labor","wage"]): return "Costs/Expenses"
    if any(k in ql for k in ["revenue","sales"]): return "Revenue"
    if any(k in ql for k in ["cash","liquidity","debt"]): return "Liquidity"
    return "Other"

def extract_metrics(q:str)->List[str]:
    ql = lower(q)
    mets = [m for m in METRIC_VOCAB if m in ql]
    norm = []
    for m in mets:
        if m in {"cost","costs"}: norm.append("costs")
        else: norm.append(m)
    return sorted(set(norm))[:8]

ENT_PROPER = re.compile(r"\b([A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,4})\b")
ROLE_HINT  = re.compile(r"(?i)\b(Investor Relations|IR|CEO|CFO|COO|CTO|Chairman|Director)\b")
GEO_HINT   = re.compile(r"(?i)\b(North America|EMEA|APAC|China|Europe|United States|US|U\.S\.)\b")

def extract_entities(q:str)->List[str]:
    ents = set()
    for m in ENT_PROPER.finditer(q):
        s = m.group(1)
        if s.lower() in {"how","what","why","does","do","the","a","an"}: continue
        ents.add(s)
    for m in ROLE_HINT.finditer(q): ents.add(m.group(1))
    for m in GEO_HINT.finditer(q):  ents.add(m.group(1))
    return list(ents)[:6]

def extract_relation(q:str)->str:
    ql = lower(q)
    for r, vocab in RELATION_LEX.items():
        if any(k in ql for k in vocab):
            return r
    return "explain" if any(k in ql for k in ["why","reason","driver"]) else "influence"

def extract_time(q:str)->Dict:
    out = {"fy": None, "quarter": None, "yoy": False, "qoq": False, "dates": []}
    mfy = FY_RE.search(q)
    if mfy:
        out["fy"] = mfy.group(1) or mfy.group(2)
    mq = Q_RE.search(q)
    if mq:
        out["quarter"] = mq.group(1)
    out["yoy"] = bool(YOY_RE.search(q))
    out["qoq"] = bool(QOQ_RE.search(q))
    out["dates"] = [m.group(0) for m in DATE_RE.finditer(q)]
    return out

def numeric_intent(q:str)->Dict:
    ql = lower(q)
    return {
        "needs_number": any(k in ql for k in ["how much","by how much","%","percent","bps","increase by","decrease by","change","growth","yoy","qoq"]),
        "yoy": bool(YOY_RE.search(q)),
        "qoq": bool(QOQ_RE.search(q)),
    }

def keywords_for_recall(q:str, ents:List[str], mets:List[str])->List[str]:
    base = []
    base += [w for w in re.findall(r"[A-Za-z]{3,}", q) if w.lower() not in {"what","how","why","does","do","the","and","for","in","of","to"}]
    base += [e for e in ents]
    base += [m for m in mets]
    return sorted(set([w.lower() for w in base]))[:12]

def preprocess_query(q:str)->Dict:
    ents = extract_entities(q)
    mets = extract_metrics(q)
    rel  = extract_relation(q)
    when = extract_time(q)
    topic= pick_topic(q)
    numi = numeric_intent(q)
    keys = keywords_for_recall(q, ents, mets)
    return {
        "raw": q,
        "topic": topic if topic in TOPIC_ENUM else "Other",
        "entities": ents,
        "metrics": mets,
        "relation": rel,
        "time_filters": when,
        "numeric_intent": numi,
        "keywords": keys
    }



