"""Day 06 pipeline entry point (run from starter/).

    python -m src.run ocr                  # OCR embedded images (Tesseract in Docker), cached
    python -m src.run verify               # read PDF (+OCR), log pages, verify benchmark evidence
    python -m src.run ingest               # chunk A/B, embed (only new/changed chunks), store
    python -m src.run retrieve --split dev # top-3 for strategies A, B, C; save traces
    python -m src.run                      # verify + ingest + retrieve --split dev

Strategy C reuses index B (identical chunks and vectors) and adds a SQL metadata
filter only for questions that carry a `filter`; this isolates the effect of
filtering (B vs C). A vs B changes only the chunker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .ingestion import annotate_sections, append_ocr_blocks, load_pdf
from .ocr import load_ocr
from .models import Chunk, DocumentPage

STARTER = Path(__file__).resolve().parents[1]
REPO = STARTER.parent
RESULTS = STARTER / "results"
MANIFEST = REPO / "data" / "manifest.json"
BENCHMARK = REPO / "benchmark" / "questions.json"
OCR_DIR = STARTER / "data" / "ocr"
OCR_RESULTS = OCR_DIR / "ocr_results.json"
# Lab requirement re-verified with the instructor: 20-30 content pages.
MIN_CONTENT_PAGES = 20

CONFIG = {
    "embedding": {
        "model": "google/embeddinggemma-2",
        "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        "batch_size": 16,
    },
    "vector_store": {"backend": "PostgreSQL + pgvector", "table": "rag_chunks",
                     "metric": "cosine (<=>)", "search": "exact (no ANN index)"},
    "top_k": 3,
    "indexes": {
        "A": {"chunker": "fixed_size", "size": 500, "overlap": 50},
        "B": {"chunker": "by_paragraph", "max_size": 800},
    },
    "strategies": {
        "A": {"index": "A", "use_filter": False,
              "description": "Fixed 500 ký tự, overlap 50, cắt tại khoảng trắng; không lọc"},
        "B": {"index": "B", "use_filter": False,
              "description": "Theo heading/đoạn, gộp đoạn tới 800 ký tự; không lọc"},
        "C": {"index": "B", "use_filter": True,
              "description": "Giống B + lọc SQL theo section khi câu hỏi có filter"},
    },
}

log = logging.getLogger("day06")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------- corpus

def load_corpus() -> tuple[list[DocumentPage], list[dict]]:
    """Load every document in the manifest; return content pages and a page log."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ocr_blocks = load_ocr(OCR_RESULTS)
    if not ocr_blocks:
        log.warning("No OCR results at %s: run `python -m src.run ocr` first", OCR_RESULTS)
    content: list[DocumentPage] = []
    page_log: list[dict] = []
    for doc in manifest["documents"]:
        path = REPO / doc["source_path_or_url"]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (see data/manifest.json)")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if doc.get("sha256") and doc["sha256"] != sha:
            log.warning("%s changed since manifest (sha256 differs): chunks will be re-synced",
                        doc["source_id"])
        pages = load_pdf(path, doc["source_id"], doc["source_path_or_url"])
        wanted = set(doc["content_pages"])
        excluded = {int(k): v for k, v in doc.get("excluded_pages", {}).items()}
        doc_content = [p for p in pages if p.page in wanted]
        annotate_sections(doc_content)
        for p in doc_content:
            append_ocr_blocks(p, ocr_blocks.get((p.document_id, p.page), []))
        for p in pages:
            m = p.metadata
            entry = {
                "document_id": p.document_id, "page": p.page, "printed_page": m["printed_page"],
                "included": p.page in wanted, "exclude_reason": excluded.get(p.page),
                "status": m["status"], "raw_chars": m["raw_chars"], "clean_chars": m["clean_chars"],
                "images": m["images"], "ocr_images": m.get("ocr_images", 0),
                "ocr_chars": m.get("ocr_chars", 0), "total_chars": len(p.text),
                "footnotes": len(m["footnotes"]), "error": m["error"],
                "headings": [s["subsection"] or s["section_title"]
                             for s in m.get("sections", []) if s.get("heading")],
            }
            page_log.append(entry)
            if p.page in wanted and m["status"] != "ok":
                log.warning("page %s p%d: status=%s (%d text chars, %d images, %d OCR chars)",
                            p.document_id, p.page, m["status"], m["clean_chars"], m["images"],
                            m.get("ocr_chars", 0))
        content.extend(doc_content)
    n_content = sum(1 for p in content if p.metadata["status"] != "empty")
    if n_content < MIN_CONTENT_PAGES:
        log.warning("Corpus has %d content pages < %d required by the lab (see data/manifest.json)",
                    n_content, MIN_CONTENT_PAGES)
    return content, page_log


def build_chunks(pages: list[DocumentPage]) -> dict[str, list[Chunk]]:
    out: dict[str, list[Chunk]] = {}
    for name, cfg in CONFIG["indexes"].items():
        if cfg["chunker"] == "fixed_size":
            out[name] = [c for p in pages for c in fixed_size(p, cfg["size"], cfg["overlap"])]
        else:
            out[name] = [c for p in pages for c in by_paragraph(p, cfg["max_size"])]
    return out


def _length_stats(values: list[int]) -> dict:
    return {"min": min(values), "mean": round(mean(values), 1), "median": median(values),
            "max": max(values), "total": sum(values)}


# ------------------------------------------------------------------------ ocr

def cmd_ocr() -> int:
    from .ocr import ocr_pdf

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    combined = None
    for doc in manifest["documents"]:
        t0 = time.perf_counter()
        result = ocr_pdf(REPO / doc["source_path_or_url"], doc["source_id"],
                         OCR_DIR / "images" / doc["source_id"])
        result["seconds"] = round(time.perf_counter() - t0, 1)
        if combined is None:
            combined = result
        else:
            combined["images"] += result["images"]
    _write_json(OCR_RESULTS, combined)
    for item in combined["images"]:
        print(f"p{item['page']:>2} img{item['image_index']} {item['variant']:>4} "
              f"conf={item['mean_conf']:>5} chars={item['chars']:>4} "
              f"{'ACCEPT' if item['accepted'] else 'skip  '} {item['reason']}")
    print(f"{sum(i['accepted'] for i in combined['images'])}/{len(combined['images'])} images "
          f"accepted -> {OCR_RESULTS.relative_to(STARTER)}")
    return 0


# --------------------------------------------------------------------- verify

def cmd_verify() -> int:
    pages, page_log = load_corpus()
    _write_json(RESULTS / "ingest_log.json", {"generated_at": _now(), "pages": page_log})

    # Fill computed page statistics into the manifest (hand-written fields are kept).
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for doc in manifest["documents"]:
        path = REPO / doc["source_path_or_url"]
        doc["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        doc["page_stats"] = [{"page": e["page"], "printed_page": e["printed_page"],
                              "included": e["included"], "status": e["status"],
                              "clean_chars": e["clean_chars"], "images": e["images"],
                              "ocr_images": e["ocr_images"], "ocr_chars": e["ocr_chars"]}
                             for e in page_log if e["document_id"] == doc["source_id"]]
        doc["content_page_count"] = len(doc["content_pages"])
        doc["content_chars"] = sum(e["clean_chars"] for e in page_log
                                   if e["document_id"] == doc["source_id"] and e["included"])
        doc["content_ocr_chars"] = sum(e["ocr_chars"] for e in page_log
                                       if e["document_id"] == doc["source_id"] and e["included"])
    _write_json(MANIFEST, manifest)

    # Verify benchmark evidence against normalized page text.
    bench = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    by_page = {(p.document_id, p.page): p for p in pages}
    sections = {s["section"] for p in pages for s in p.metadata["sections"] if s["section"]}
    results, problems = [], []
    for q in bench["questions"]:
        golds = [{"gold_source_id": q["gold_source_id"], "gold_page": q["gold_page"],
                  "evidence_quote": q["evidence_quote"]}] + q.get("alternatives", [])
        checks = []
        for g in golds:
            page = by_page.get((g["gold_source_id"], g["gold_page"]))
            found = bool(page) and _collapse(g["evidence_quote"]) in _collapse(page.text)
            checks.append({**g, "page_indexed": bool(page), "quote_found": found})
            if not found:
                problems.append(f"{q['id']}: evidence not found on p{g['gold_page']}")
        filt = q.get("filter")
        filter_ok = None
        if filt:
            value = filt.get("section")
            page = by_page.get((q["gold_source_id"], q["gold_page"]))
            page_sections = {s["section"] for s in page.metadata["sections"]} if page else set()
            filter_ok = value in sections and value in page_sections
            if not filter_ok:
                problems.append(f"{q['id']}: filter section={value} not valid for gold page")
        results.append({"id": q["id"], "split": q["split"], "type": q["type"],
                        "evidence": checks, "filter": filt, "filter_valid": filter_ok})
    splits = {s: sum(1 for q in bench["questions"] if q["split"] == s) for s in ("dev", "test")}
    n_filter = {s: sum(1 for q in bench["questions"] if q["split"] == s and q.get("filter"))
                for s in ("dev", "test")}
    rules = {
        ">=20 questions": len(bench["questions"]) >= 20,
        ">=15 dev": splits["dev"] >= 15, ">=5 test": splits["test"] >= 5,
        ">=3 filter questions": sum(n_filter.values()) >= 3, ">=1 test filter": n_filter["test"] >= 1,
    }
    problems += [f"rule failed: {k}" for k, ok in rules.items() if not ok]
    report = {"generated_at": _now(), "questions": len(bench["questions"]), "splits": splits,
              "filter_questions": n_filter, "rules": rules, "section_values": sorted(sections),
              "problems": problems, "details": results}
    _write_json(RESULTS / "benchmark_verification.json", report)
    included = [e for e in page_log if e["included"]]
    print(f"Pages: {len(page_log)} physical, {len(included)} included "
          f"({sum(e['status'] == 'ok' for e in included)} ok, "
          f"{sum(e['status'] == 'low_text' for e in included)} low_text, "
          f"{sum(e['status'] == 'empty' for e in included)} empty)")
    print(f"Benchmark: {len(bench['questions'])} questions {splits}, filter {n_filter}, "
          f"evidence found {sum(all(c['quote_found'] for c in r['evidence']) for r in results)}"
          f"/{len(results)}")
    for p in problems:
        print("  PROBLEM:", p)
    return 1 if problems else 0


# --------------------------------------------------------------------- ingest

def _open_store_and_embedder():
    from .embeddings import MyEmbedder
    from .store import PgVectorStore

    load_dotenv(STARTER / ".env")
    model = os.getenv("EMBEDDING_MODEL") or CONFIG["embedding"]["model"]
    if model != CONFIG["embedding"]["model"]:
        raise SystemExit(f"EMBEDDING_MODEL={model} differs from the benchmark config; "
                         "keep one model for all strategies")
    t0 = time.perf_counter()
    embedder = MyEmbedder(model, revision=CONFIG["embedding"]["revision"],
                          batch_size=CONFIG["embedding"]["batch_size"])
    load_s = time.perf_counter() - t0
    store = PgVectorStore(os.getenv("DATABASE_URL", ""), embedder.dimensions)
    store.create_schema()
    return store, embedder, load_s


def cmd_ingest() -> int:
    from .store import content_hash

    started = _now()
    t0 = time.perf_counter()
    pages, _ = load_corpus()
    load_pdf_s = time.perf_counter() - t0
    store, embedder, model_load_s = _open_store_and_embedder()
    model_name = embedder.model_name

    t0 = time.perf_counter()
    chunks_by_index = build_chunks(pages)
    chunk_s = time.perf_counter() - t0

    per_index = {}
    for name, chunks in chunks_by_index.items():
        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate chunk IDs in index {name}")
        _write_jsonl(RESULTS / f"chunks_{name}.jsonl", [{
            "chunk_id": c.chunk_id, "document_id": c.document_id, "source": c.source,
            "page": c.page, "chars": len(c.text), "metadata": c.metadata, "text": c.text,
        } for c in chunks])

        t_embed = time.perf_counter()
        changed, deleted = [], 0
        for doc_id in sorted({c.document_id for c in chunks}):
            doc_chunks = [c for c in chunks if c.document_id == doc_id]
            existing = store.existing_hashes(name, doc_id)
            changed += [c for c in doc_chunks
                        if existing.get(c.chunk_id) != content_hash(c, model_name)]
            deleted += store.delete_obsolete(name, doc_id, [c.chunk_id for c in doc_chunks])
        vectors = embedder.embed_documents([c.text for c in changed])
        embed_s = time.perf_counter() - t_embed
        t_store = time.perf_counter()
        written = store.upsert(name, changed, vectors, model_name) if changed else 0
        store_s = time.perf_counter() - t_store

        lengths = [len(c.text) for c in chunks]
        tokens = embedder.count_tokens([c.text for c in chunks])
        per_index[name] = {
            "config": CONFIG["indexes"][name],
            "chunks": len(chunks),
            "chars": _length_stats(lengths),
            "tokens": _length_stats(tokens),
            "embedded_this_run": len(changed),
            "upserted": written,
            "deleted_obsolete": deleted,
            "embed_seconds": round(embed_s, 2),
            "store_seconds": round(store_s, 2),
        }
        print(f"[{name}] {len(chunks)} chunks | embedded {len(changed)} in {embed_s:.1f}s | "
              f"upserted {written} | deleted {deleted}")

    counts = store.counts()
    for name, c in counts.items():
        if c["rows"] != c["distinct_ids"] or c["rows"] != c["distinct_texts"]:
            log.warning("index %s: possible duplicates %s", name, c)
    run = {
        "started_at": started,
        "embedding": embedder.describe(),
        "vector_store": {**CONFIG["vector_store"], **store.versions()},
        "timing_seconds": {"load_pdf": round(load_pdf_s, 2), "model_load": round(model_load_s, 2),
                           "chunking": round(chunk_s, 3)},
        "indexes": per_index,
        "db_counts_after": counts,
        "cost_usd": 0.0,
    }
    report_path = RESULTS / "ingest_report.json"
    history = []
    if report_path.exists():
        history = json.loads(report_path.read_text(encoding="utf-8")).get("runs", [])
    history.append(run)
    _write_json(report_path, {"config": CONFIG, "runs": history})
    print("DB rows per index:", counts)
    return 0


# ------------------------------------------------------------------- retrieve

def cmd_retrieve(split: str) -> int:
    if split in ("test", "all") and not (RESULTS / "selection.json").exists():
        raise SystemExit("Run `evaluate --split dev` and `select` before touching the test split")
    store, embedder, _ = _open_store_and_embedder()
    bench = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    questions = [q for q in bench["questions"] if split == "all" or q["split"] == split]
    traces = []
    for q in questions:
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["question"])
        embed_ms = (time.perf_counter() - t0) * 1000
        for strategy, cfg in CONFIG["strategies"].items():
            filters = q.get("filter") if cfg["use_filter"] else None
            t1 = time.perf_counter()
            hits = store.search(cfg["index"], qvec, CONFIG["top_k"], filters)
            search_ms = (time.perf_counter() - t1) * 1000
            traces.append({
                "strategy": strategy, "index": cfg["index"], "question_id": q["id"],
                "split": q["split"], "type": q["type"], "question": q["question"],
                "filter_applied": filters, "query_embed_ms": round(embed_ms, 1),
                "search_ms": round(search_ms, 1),
                "gold": {"source_id": q["gold_source_id"], "page": q["gold_page"],
                         "evidence_quote": q["evidence_quote"],
                         "alternatives": q.get("alternatives", [])},
                "top_k": [{"rank": i + 1, **h} for i, h in enumerate(hits)],
            })
    out = RESULTS / f"traces_{split}.jsonl"
    _write_jsonl(out, traces)
    _write_json(RESULTS / "run_config.json", {"generated_at": _now(), "config": CONFIG,
                                               "embedding": embedder.describe()})
    print(f"Saved {len(traces)} traces ({len(questions)} questions x "
          f"{len(CONFIG['strategies'])} strategies) to {out.relative_to(STARTER)}")
    for t in traces:
        pages = ",".join(f"p{h['page']}" for h in t["top_k"])
        filt = f" filter={t['filter_applied']}" if t["filter_applied"] else ""
        print(f"  {t['question_id']} [{t['strategy']}] top3 pages: {pages}{filt}")
    return 0


# ------------------------------------------------------------------- evaluate

AUDIT = RESULTS / "audit"
SELECTION = RESULTS / "selection.json"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} missing; run the previous step first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cmd_evaluate(split: str) -> int:
    import csv

    from .evaluation import evaluate_query, summarize

    bench = {q["id"]: q for q in json.loads(BENCHMARK.read_text(encoding="utf-8"))["questions"]}
    traces = _load_jsonl(RESULTS / f"traces_{split}.jsonl")
    evals = []
    for t in traces:
        e = evaluate_query(bench[t["question_id"]], t["top_k"])
        evals.append({"strategy": t["strategy"], **e})
    summary = summarize(evals)
    _write_json(RESULTS / f"eval_{split}.json",
                {"generated_at": _now(), "split": split, "summary": summary, "per_query": evals})
    with (RESULTS / f"eval_{split}_per_query.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "question_id", "split", "type", "has_filter", "hit_at_1",
                    "hit_at_3", "first_hit_rank", "page_hit_at_3", "evidence_split_suspected",
                    "filter_compliant", "top3_pages", "top3_chunk_ids"])
        for e in evals:
            w.writerow([e["strategy"], e["question_id"], e["split"], e["type"], e["has_filter"],
                        int(e["hit_at_1"]), int(e["hit_at_3"]), e["first_hit_rank"] or "",
                        int(e["page_hit_at_3"]), int(e["evidence_split_suspected"]),
                        "" if e["filter_compliant"] is None else int(e["filter_compliant"]),
                        " ".join(str(r["page"]) for r in e["ranked"]),
                        " ".join(r["chunk_id"] for r in e["ranked"])])
    print(f"{'strategy':<9}{'split':<6}{'Hit@1':>8}{'EvHit@3':>9}{'PageHit@3':>11}"
          f"{'MRR@3':>7}{'split?':>7}{'filter':>8}  misses")
    for strategy, by_split in summary.items():
        for sp, m in by_split.items():
            filt = f"{m['filter_compliant']}/{m['filter_questions']}" if m["filter_questions"] else "N/A"
            print(f"{strategy:<9}{sp:<6}{m['hit_at_1']:>4}/{m['N']:<3}{m['evidence_hit_at_3']:>5}/"
                  f"{m['N']:<3}{m['page_hit_at_3']:>7}/{m['N']:<3}{m['mrr_at_3']:>7}"
                  f"{m['evidence_split_suspected']:>7}{filt:>8}  {','.join(m['misses'])}")
    return 0


def cmd_select() -> int:
    """Apply the criterion frozen in benchmark/questions.json to DEV results only."""
    bench = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    dev = json.loads((RESULTS / "eval_dev.json").read_text(encoding="utf-8"))["summary"]
    report = json.loads((RESULTS / "ingest_report.json").read_text(encoding="utf-8"))
    chunks = {k: v["chunks"] for k, v in report["runs"][-1]["indexes"].items()}

    def key(name: str):
        m = dev[name]["dev"]
        index = CONFIG["strategies"][name]["index"]
        # Higher is better for the first two; fewer chunks = fewer resources.
        return (m["evidence_hit_at_3"], m["hit_at_1"], -chunks[index])

    ranking = sorted(dev, key=key, reverse=True)
    chosen = ranking[0]
    tied = [s for s in ranking if key(s) == key(chosen)]
    result = {
        "selected_at": _now(),
        "criterion": bench["selection_criterion"],
        "dev_metrics": {s: {k: dev[s]["dev"][k] for k in ("N", "hit_at_1", "evidence_hit_at_3",
                                                         "filter_compliant", "filter_questions")}
                        for s in dev},
        "chunks_per_index": chunks,
        "ranking": ranking,
        "selected": chosen,
        "full_ties": tied if len(tied) > 1 else [],
        "note": ("Chọn trước khi chạy test. Bậc 3 của tiêu chí (grounding thủ công) chỉ được chấm "
                 "trên test nên không dùng để chọn; khi hòa Hit@3 và Hit@1 trên dev thì xét tài "
                 "nguyên (số chunk). Nếu vẫn hòa (B và C dùng chung index), ưu tiên chiến lược "
                 "đứng trước trong thứ tự sắp xếp ổn định A, B, C."),
    }
    _write_json(SELECTION, result)
    print(json.dumps({k: result[k] for k in ("dev_metrics", "chunks_per_index", "ranking",
                                              "selected", "full_ties")},
                     ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------- audit

def _page_body(page: DocumentPage) -> str:
    ocr_offset = page.metadata.get("ocr_offset")
    return page.text if ocr_offset is None else page.text[:ocr_offset - 1]


def cmd_audit_samples() -> int:
    """Write samples for manual coherence / metadata / grounding labelling."""
    import csv

    pages, _ = load_corpus()
    by_page = {(p.document_id, p.page): p for p in pages}
    AUDIT.mkdir(parents=True, exist_ok=True)

    # Coherence: 10 positions fixed before scoring = midpoint of the PDF text of
    # the first 10 content pages (sorted by page); same positions for every index.
    positions = []
    for p in sorted(pages, key=lambda x: (x.document_id, x.page))[:10]:
        body = _page_body(p)
        off = len(body) // 2
        positions.append((p, off, body[max(0, off - 60):off + 60].replace("\n", " ")))
    rows = []
    for name in CONFIG["indexes"]:
        chunks = _load_jsonl(RESULTS / f"chunks_{name}.jsonl")
        for i, (p, off, snippet) in enumerate(positions, start=1):
            cands = [c for c in chunks if c["document_id"] == p.document_id and c["page"] == p.page
                     and c["metadata"]["char_start"] <= off < c["metadata"]["char_end"]]
            c = min(cands, key=lambda c: c["metadata"]["chunk_index"])
            rows.append({"position_id": f"pos{i:02d}", "page": p.page, "offset": off,
                         "source_snippet": snippet, "index": name, "chunk_id": c["chunk_id"],
                         "chunk_chars": c["chars"], "chunk_text": c["text"]})
    with (AUDIT / "coherence_samples.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Metadata + grounding: every test question x strategy x top-3.
    bench = {q["id"]: q for q in json.loads(BENCHMARK.read_text(encoding="utf-8"))["questions"]}
    traces = _load_jsonl(RESULTS / "traces_test.jsonl")
    meta_rows, md = [], ["# Grounding samples (test split)\n"]
    for t in traces:
        q = bench[t["question_id"]]
        md.append(f"\n## {t['question_id']} [{t['strategy']}] {q['question']}\n\n")
        md.append(f"- Filter: {t['filter_applied']}\n- Expected: {q['expected_answer']}\n"
                  f"- Gold: p{q['gold_page']}: \"{q['evidence_quote']}\"\n")
        for h in t["top_k"]:
            page = by_page[(h["document_id"], h["page"])]
            sections = {s["section"] for s in page.metadata["sections"]}
            meta = h["metadata"]
            meta_rows.append({
                "question_id": t["question_id"], "strategy": t["strategy"], "rank": h["rank"],
                "chunk_id": h["chunk_id"], "source": h["source"], "page": h["page"],
                "printed_page": meta.get("printed_page"), "section": h["section"],
                "subsection": meta.get("subsection"), "content_source": meta.get("content_source"),
                "source_file_exists": (REPO / h["source"]).exists(),
                "text_found_on_page": _collapse(h["text"]) in _collapse(page.text),
                "section_on_page": h["section"] in sections,
                "printed_page_ok": meta.get("printed_page") == page.metadata.get("printed_page"),
                "filter": json.dumps(q.get("filter"), ensure_ascii=False) if q.get("filter") else "",
                "matches_filter": (all(h.get(k) == v for k, v in q["filter"].items())
                                   if q.get("filter") else ""),
            })
            text = h["text"].replace("\n", "\n> ")
            md.append(f"\n**#{h['rank']}** `{h['chunk_id']}` p{h['page']} {h['section']} "
                      f"score={h['score']}\n\n> {text}\n")
    with (AUDIT / "metadata_samples.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta_rows[0]))
        w.writeheader()
        w.writerows(meta_rows)
    (AUDIT / "grounding_samples.md").write_text("".join(md), encoding="utf-8")
    print(f"Wrote {len(rows)} coherence, {len(meta_rows)} metadata samples and "
          f"{len(traces)} grounding cases to {AUDIT.relative_to(STARTER)}")
    return 0


def cmd_audit_summary() -> int:
    """Aggregate manual labels (results/audit/*_labels.csv) per strategy."""
    import csv

    def read(name: str) -> list[dict]:
        path = AUDIT / name
        if not path.exists():
            raise SystemExit(f"{path} missing: label the samples first")
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    coherence = read("coherence_labels.csv")
    metadata = read("metadata_labels.csv")
    grounding = read("grounding_labels.csv")
    test_eval = json.loads((RESULTS / "eval_test.json").read_text(encoding="utf-8"))["summary"]
    out = {"generated_at": _now(),
           "labeler": sorted({r["labeler"] for r in coherence + metadata + grounding}),
           "strategies": {}}
    for strategy, cfg in CONFIG["strategies"].items():
        coh = [int(r["label"]) for r in coherence if r["index"] == cfg["index"]]
        meta = [r for r in metadata if r["strategy"] == strategy]
        gr = [int(r["label"]) for r in grounding if r["strategy"] == strategy]
        m = test_eval[strategy]["test"]
        out["strategies"][strategy] = {
            "coherence": {"n": len(coh), "mean_of_2": round(mean(coh), 2),
                          "counts": {str(k): coh.count(k) for k in (0, 1, 2)},
                          "note": "C dùng chung chunk với B" if cfg["index"] != strategy else ""},
            "metadata_correct": {"n": sum(r["correct"] == "1" for r in meta), "N": len(meta),
                                 "errors": [f"{r['question_id']}#{r['rank']}: {r['note']}"
                                            for r in meta if r["correct"] != "1"]},
            "filter_compliance_test": (f"{m['filter_compliant']}/{m['filter_questions']}"
                                       if m["filter_questions"] else "N/A"),
            "grounding": {"n": len(gr), "counts": {str(k): gr.count(k) for k in (0, 1, 2)},
                          "mean_of_2": round(mean(gr), 2)},
        }
    _write_json(RESULTS / "audit_summary.json", out)
    print(json.dumps(out["strategies"], ensure_ascii=False, indent=1))
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ocr")
    sub.add_parser("verify")
    sub.add_parser("ingest")
    r = sub.add_parser("retrieve")
    r.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    e = sub.add_parser("evaluate")
    e.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    sub.add_parser("select")
    sub.add_parser("audit-samples")
    sub.add_parser("audit-summary")
    args = parser.parse_args()
    if args.cmd == "ocr":
        raise SystemExit(cmd_ocr())
    if args.cmd == "verify":
        raise SystemExit(cmd_verify())
    if args.cmd == "ingest":
        raise SystemExit(cmd_ingest())
    if args.cmd == "retrieve":
        raise SystemExit(cmd_retrieve(args.split))
    if args.cmd == "evaluate":
        raise SystemExit(cmd_evaluate(args.split))
    if args.cmd == "select":
        raise SystemExit(cmd_select())
    if args.cmd == "audit-samples":
        raise SystemExit(cmd_audit_samples())
    if args.cmd == "audit-summary":
        raise SystemExit(cmd_audit_summary())
    code = (cmd_verify() or cmd_ingest() or cmd_retrieve("dev") or cmd_evaluate("dev")
            or cmd_select() or cmd_retrieve("test") or cmd_evaluate("test")
            or cmd_audit_samples())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
