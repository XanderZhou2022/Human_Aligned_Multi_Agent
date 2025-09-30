#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
card.py — Core data structure for evidence "cards"

This module provides a compact, dependency-free implementation of the evidence
"Card" concept used for transparent, auditable retrieval over long corporate reports.
It focuses ONLY on the core representation and basic utilities (validation,
normalization, JSON (de)serialization, deduplication key, and a lightweight
compatibility check). Non-core elements (training, retrieval, scoring, pipelines)
are intentionally excluded.

Key ideas:
  - A Card records the minimum auditable facts: entity, metric, period, number (optional),
    a verbatim span, and where that span came from (document + location).
  - All fields are immutable once constructed to preserve provenance.
  - Normalization helpers keep period and numbers in a canonical form to reduce
    temporal / numeric mismatches downstream.

No external dependencies beyond the Python Standard Library.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Optional, Dict, Any
import json
import re

# -----------------------------
# Helpers (minimal, dependency-free)
# -----------------------------

_WS = re.compile(r"\s+")

def _norm_space(s: str) -> str:
    """Collapse consecutive whitespace into single spaces and strip ends."""
    return _WS.sub(" ", s).strip()

def _norm_case(s: str) -> str:
    """Lowercase for case-insensitive comparisons (without altering stored values)."""
    return s.lower()

# --- Period normalization ---
_FY = re.compile(r"^(?:fy|f\.?y\.?|fiscal\s*year)\s*(\d{4})$", re.I)
_QFY = re.compile(r"^q(\d)\s*(?:fy|f\.?y\.?|fiscal\s*year)?\s*(\d{4})$", re.I)
_CALQ = re.compile(r"^q(\d)\s*(\d{4})$", re.I)  # e.g., Q2 2024

def normalize_period(raw: str) -> str:
    """Normalize period labels to a compact canonical form.

    Rules (examples):
      - "FY 2023" -> "FY2023"
      - "Q2 FY2024" or "Q2 2024" -> "Q2FY2024"
      - If not matched, collapse spaces and return as-is (caller can still audit).
    """
    s = _norm_space(raw)
    m = _FY.match(s)
    if m:
        return f"FY{m.group(1)}"
    m = _QFY.match(s)
    if m:
        q, y = m.groups()
        return f"Q{q}FY{y}"
    m = _CALQ.match(s)
    if m:
        q, y = m.groups()
        return f"Q{q}FY{y}"
    return s  # fallback, still human-readable

# --- Number parsing ---
_NUM_CLEAN = re.compile(r"[ ,\u00A0]")  # spaces, commas, NBSP

def parse_number(raw: Optional[str]) -> Optional[Decimal]:
    """Parse a financial number string into Decimal or return None.
    Handles:
      - commas / thin spaces: "1,234.50" -> 1234.50
      - parentheses for negatives: "(12.3)" -> -12.3
      - percents are preserved as scalar (caller handles unit="%")
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = _NUM_CLEAN.sub("", s)
    # Allow trailing % but ignore for numeric scalar (unit should capture it)
    if s.endswith("%"):
        s = s[:-1]
    try:
        val = Decimal(s)
    except InvalidOperation:
        return None
    return (-val if neg else val).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)

# -----------------------------
# Core data types
# -----------------------------

@dataclass(frozen=True)
class SpanLoc:
    """Where the span was found. All fields are optional to keep the type permissive.
    doc_id: identifier such as a filename, URL, or database key
    page: page number (1-based if PDF-like), None if not applicable
    start: character offset within the page/section
    end: character offset within the page/section (exclusive)
    """
    doc_id: Optional[str] = None
    page: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None

@dataclass(frozen=True)
class Card:
    """Minimal, immutable record of evidence.

    Fields
    ------
    entity: str
        The business entity the fact belongs to (issuer, subsidiary, segment, etc.).
    metric: str
        The financial concept (e.g., revenue, operating income, EPS).
    period: str
        Fiscal label, normalized via `normalize_period` (e.g., "Q2FY2024", "FY2023").
    span_text: str
        Verbatim snippet supporting the evidence.
    number: Optional[Decimal]
        Parsed numeric value if an explicit number exists; None if text-only.
    unit: Optional[str]
        Unit label such as "USD", "USDm", "%". Kept as-is for auditability.
    span_loc: Optional[SpanLoc]
        Where the snippet came from (document + page/offsets).
    source: Optional[str]
        Disclosure source type (e.g., "10-K", "10-Q", "Press Release").

    Design
    ------
    - Immutable dataclass to safeguard provenance once created.
    - Validation ensures key fields are non-empty and within reasonable length.
    - `dedup_key()` produces a stable, coarse-grained key for near-duplicate collapse.
    - `to_json()` / `from_json()` provide a stable interchange format.
    """
    entity: str
    metric: str
    period: str
    span_text: str
    number: Optional[Decimal] = None
    unit: Optional[str] = None
    span_loc: Optional[SpanLoc] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:  # type: ignore[override]
        # Normalize and validate without mutating stored values (immutability preserved)
        if not self.entity or not self.metric or not self.period or not self.span_text:
            raise ValueError("entity, metric, period, and span_text are required")
        if len(self.entity) > 200 or len(self.metric) > 200:
            raise ValueError("entity/metric too long; keep <= 200 chars for auditability")
        if len(self.span_text) > 5000:
            raise ValueError("span_text too long; keep <= 5000 chars")
        # Period should already be normalized by factory, but guard here too.
        p = normalize_period(self.period)
        if p != self.period:
            # Users should construct via `Card.make(...)` to get normalized period.
            raise ValueError("period must be canonical (use Card.make to normalize)")

    # --------------
    # Constructors
    # --------------
    @staticmethod
    def make(*, entity: str, metric: str, period: str, span_text: str,
             number: Optional[str] = None, unit: Optional[str] = None,
             span_loc: Optional[SpanLoc] = None, source: Optional[str] = None) -> "Card":
        """Factory creating a Card with canonical period and parsed number.
        `number` may be a human string; it will be parsed to Decimal when possible.
        """
        period_norm = normalize_period(period)
        num_val = parse_number(number)
        return Card(
            entity=_norm_space(entity),
            metric=_norm_space(metric),
            period=period_norm,
            span_text=_norm_space(span_text),
            number=num_val,
            unit=(unit.strip() if unit else None),
            span_loc=span_loc,
            source=(source.strip() if source else None),
        )

    # --------------
    # Utilities
    # --------------
    def dedup_key(self) -> str:
        """A coarse, stable key for collapsing near-duplicates.
        - case-insensitive entity/metric/period
        - numeric value if present rounded to 4dp; else "none"
        - ignores span_text to avoid micro-duplicates from minor OCR changes
        """
        num = str(self.number) if self.number is not None else "none"
        return "|".join([
            _norm_case(self.entity),
            _norm_case(self.metric),
            _norm_case(self.period),
            num,
            self.unit or "",
            _norm_case(self.source) if self.source else "",
        ])

    def to_json(self) -> str:
        """Serialize to JSON string (Decimal -> string for stability)."""
        payload: Dict[str, Any] = {
            "entity": self.entity,
            "metric": self.metric,
            "period": self.period,
            "span_text": self.span_text,
            "number": (str(self.number) if self.number is not None else None),
            "unit": self.unit,
            "span_loc": (
                {
                    "doc_id": self.span_loc.doc_id,
                    "page": self.span_loc.page,
                    "start": self.span_loc.start,
                    "end": self.span_loc.end,
                }
                if self.span_loc
                else None
            ),
            "source": self.source,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def from_json(data: str) -> "Card":
        """Inverse of `to_json`. Accepts numbers serialized as strings."""
        obj = json.loads(data)
        loc = obj.get("span_loc")
        span_loc = None
        if isinstance(loc, dict):
            span_loc = SpanLoc(
                doc_id=loc.get("doc_id"),
                page=loc.get("page"),
                start=loc.get("start"),
                end=loc.get("end"),
            )
        # Use factory to reapply normalization/parsing
        return Card.make(
            entity=obj["entity"],
            metric=obj["metric"],
            period=obj["period"],
            span_text=obj["span_text"],
            number=obj.get("number"),
            unit=obj.get("unit"),
            span_loc=span_loc,
            source=obj.get("source"),
        )

    def explain(self) -> str:
        """Human-friendly single-line rationale summary for audits and logs."""
        parts = [
            f"entity={self.entity}",
            f"metric={self.metric}",
            f"period={self.period}",
            f"number={self.number}{self.unit or '' if self.number is not None else ''}",
            f"source={self.source or 'N/A'}",
        ]
        if self.span_loc and self.span_loc.doc_id:
            parts.append(
                f"loc={self.span_loc.doc_id}#p{self.span_loc.page or '?'}:{self.span_loc.start or '?'}-{self.span_loc.end or '?'}"
            )
        return " | ".join(parts)

    # --------------
    # Lightweight intent compatibility (optional helper)
    # --------------
    def is_compatible(
        self,
        *,
        required_metrics: Optional[set[str]] = None,
        required_periods: Optional[set[str]] = None,
        required_entities: Optional[set[str]] = None,
        need_number: bool = False,
    ) -> bool:
        """Check whether this card satisfies a minimal intent.
        Matching is exact on normalized fields (no alias expansion here).
        """
        if required_metrics and _norm_case(self.metric) not in { _norm_case(m) for m in required_metrics }:
            return False
        if required_periods and _norm_case(self.period) not in { _norm_case(p) for p in required_periods }:
            return False
        if required_entities and _norm_case(self.entity) not in { _norm_case(e) for e in required_entities }:
            return False
        if need_number and self.number is None:
            return False
        return True

__all__ = ["Card", "SpanLoc", "normalize_period", "parse_number"]
