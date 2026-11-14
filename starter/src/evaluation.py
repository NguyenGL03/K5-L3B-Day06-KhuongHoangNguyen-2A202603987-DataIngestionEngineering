"""Evaluate retrieved evidence and preserve per-query traces.

Definition (GUIDE.md §4.1): a retrieved chunk is a *hit* when it comes from a
gold (source, page) AND contains the full recorded evidence quote (whitespace-
insensitive). Hit@1 = questions whose top-1 is a hit / questions; Evidence
Hit@3 = questions with at least one hit in the top-3 / questions.

Chunk coherence and grounding are NOT inferred here: they are manual labels
(see results/audit/). Evidence split across chunks is only flagged.
"""
from __future__ import annotations

import re
from typing import Any


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _golds(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Primary gold plus documented alternatives (any one is accepted)."""
    main = {"source_id": question["gold_source_id"], "page": question["gold_page"],
            "evidence_quote": question["evidence_quote"]}
    alts = [{"source_id": a.get("gold_source_id", question["gold_source_id"]),
             "page": a["gold_page"], "evidence_quote": a["evidence_quote"]}
            for a in question.get("alternatives", [])]
    return [main, *alts]


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and full gold evidence per hit; keep hit@1/hit@3 trace."""
    golds = _golds(question)
    filt = question.get("filter") or {}
    ranked = []
    for rank, hit in enumerate(hits, start=1):
        text = _collapse(hit["text"])
        hit_tokens = set(_tokens(hit["text"]))
        page_match = any(hit["document_id"] == g["source_id"] and hit["page"] == g["page"]
                         for g in golds)
        full = any(hit["document_id"] == g["source_id"] and hit["page"] == g["page"]
                   and _collapse(g["evidence_quote"]) in text for g in golds)
        recall = max((sum(t in hit_tokens for t in _tokens(g["evidence_quote"]))
                      / max(1, len(_tokens(g["evidence_quote"]))) for g in golds), default=0.0)
        ranked.append({
            "rank": rank, "chunk_id": hit["chunk_id"], "page": hit["page"],
            "section": hit.get("section"), "score": hit.get("score"),
            "page_match": page_match, "evidence_full": full,
            "evidence_token_recall": round(recall, 3),
            "is_hit": page_match and full,
            "matches_filter": (all(hit.get(k) == v for k, v in filt.items()) if filt else None),
        })
    first = next((r["rank"] for r in ranked if r["is_hit"]), None)
    # Evidence split: gold page retrieved, no single chunk holds the full quote,
    # but the gold-page chunks of the top-3 together contain every quote token.
    gold_page_tokens = set()
    for r, hit in zip(ranked, hits):
        if r["page_match"]:
            gold_page_tokens |= set(_tokens(hit["text"]))
    quote_tokens = set(_tokens(golds[0]["evidence_quote"]))
    split = (first is None and any(r["page_match"] for r in ranked)
             and quote_tokens <= gold_page_tokens)
    return {
        "question_id": question["id"], "split": question["split"], "type": question["type"],
        "has_filter": bool(filt),
        "hit_at_1": bool(ranked) and ranked[0]["is_hit"],
        "hit_at_3": first is not None,
        "first_hit_rank": first,
        "page_hit_at_3": any(r["page_match"] for r in ranked),
        "evidence_split_suspected": split,
        "filter_compliant": (all(r["matches_filter"] for r in ranked) if filt else None),
        "ranked": ranked,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split (n/N)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in traces:
        groups.setdefault((t["strategy"], t["split"]), []).append(t)
    out: dict[str, Any] = {}
    for (strategy, split), rows in sorted(groups.items()):
        n = len(rows)
        filt = [r for r in rows if r["has_filter"]]
        hit1 = sum(r["hit_at_1"] for r in rows)
        hit3 = sum(r["hit_at_3"] for r in rows)
        out.setdefault(strategy, {})[split] = {
            "N": n,
            "hit_at_1": hit1, "hit_at_1_rate": round(hit1 / n, 3),
            "evidence_hit_at_3": hit3, "evidence_hit_at_3_rate": round(hit3 / n, 3),
            "page_hit_at_3": sum(r["page_hit_at_3"] for r in rows),
            "mrr_at_3": round(sum(1 / r["first_hit_rank"] for r in rows if r["first_hit_rank"]) / n, 3),
            "evidence_split_suspected": sum(r["evidence_split_suspected"] for r in rows),
            "filter_questions": len(filt),
            "filter_compliant": sum(bool(r["filter_compliant"]) for r in filt) if filt else None,
            "misses": [r["question_id"] for r in rows if not r["hit_at_3"]],
        }
    return out
