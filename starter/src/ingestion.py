"""Read and normalize source data, keeping every line traceable to its PDF page."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

# Characters that pdfTeX output may contain but that carry no meaning for retrieval.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_SPACES = re.compile(r"[ \t  -   　]+")

# A footnote line starts with its number glued to the text, e.g. "2Công ty Cổ phần ...".
_FOOTNOTE_START = re.compile(r"^(\d{1,2})(?=[^\d\s.,/:%])")
_CHAPTER = re.compile(r"^(\d{1,2})\s+(\S.{1,78})$")
_SUBSECTION = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})\s+(\S.{1,88})$")


def normalize_text(text: str) -> str:
    """Normalize Unicode (NFC) and whitespace while keeping line structure.

    Line breaks are kept because they carry list/heading structure that the
    paragraph chunker needs; only runs of spaces and blank lines are collapsed.
    """
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    for ligature, plain in _LIGATURES.items():
        text = text.replace(ligature, plain)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _boilerplate_key(line: str) -> str:
    """Key used to detect running headers/footers: digits (page numbers) masked."""
    return re.sub(r"\d+", "#", line.strip())


def _find_boilerplate(pages_lines: list[list[str]], min_share: float = 0.6) -> set[str]:
    """Lines repeated at the top (first 3) or bottom (last 2) of most pages."""
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        edge = {_boilerplate_key(line) for line in lines[:3] + lines[-2:] if line.strip()}
        counts.update(edge)
    threshold = max(2, int(len(pages_lines) * min_share))
    return {key for key, n in counts.items() if n >= threshold}


def _split_footnotes(lines: list[str]) -> tuple[list[str], list[str]]:
    """Move a trailing footnote block (in the last lines of a page) out of body text."""
    for i in range(max(0, len(lines) - 4), len(lines)):
        match = _FOOTNOTE_START.match(lines[i])
        if not match:
            continue
        marker = match.group(1)
        body, notes = lines[:i], lines[i:]
        # Remove the matching superscript marker glued to the end of a body line,
        # e.g. "... tích cực.1" -> "... tích cực."
        marker_re = re.compile(rf"(?<=[.,;%)]){marker}$")
        for j in range(len(body) - 1, -1, -1):
            if marker_re.search(body[j]):
                body[j] = marker_re.sub("", body[j])
                break
        return body, notes
    return lines, []


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep page number, source and an extraction log.

    Running headers/footers are removed (detected as lines repeated on >=60% of
    pages), footnotes are moved to metadata. Pages with no or little text are
    flagged in metadata; OCR is not performed here.
    """
    reader = PdfReader(str(path))
    raw_pages: list[tuple[str, int]] = []
    errors: dict[int, str] = {}
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # keep going, but record the failure
            raw, errors[number] = "", f"{type(exc).__name__}: {exc}"
        try:
            n_images = len(page.images)
        except Exception:
            n_images = -1
        raw_pages.append((raw, n_images))

    pages_lines = [normalize_text(raw).split("\n") for raw, _ in raw_pages]
    boilerplate = _find_boilerplate(pages_lines)

    pages: list[DocumentPage] = []
    for number, ((raw, n_images), lines) in enumerate(zip(raw_pages, pages_lines), start=1):
        removed = [l for l in lines if _boilerplate_key(l) in boilerplate]
        body = [l for l in lines if l and _boilerplate_key(l) not in boilerplate]
        body, footnotes = _split_footnotes(body)
        text = "\n".join(body)
        printed = None
        for line in removed:
            m = re.search(r"Trang (\d+)/(\d+)$", line)
            if m:
                printed = int(m.group(1))
        status = "ok"
        if not text.strip():
            status = "empty"
        elif len(text) < 300:
            status = "low_text"
        pages.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=number,
            text=text,
            metadata={
                "printed_page": printed,
                "raw_chars": len(raw),
                "clean_chars": len(text),
                "images": n_images,
                "status": status,
                "error": errors.get(number),
                "removed_boilerplate_lines": len(removed),
                "footnotes": footnotes,
            },
        ))
    return pages


def slugify(title: str) -> str:
    """ASCII slug from a heading taken verbatim from the source ("Quản lý rủi ro" -> "quan_ly_rui_ro")."""
    title = title.replace("Đ", "D").replace("đ", "d")
    ascii_ = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_.lower()).strip("_")


def annotate_sections(pages: list[DocumentPage]) -> None:
    """Attach section spans to each page from numbered headings found in the text.

    Chapters must appear in order (1, 2, 3, ...) and subsections must belong to
    the current chapter; this avoids mistaking wrapped numbers for headings.
    Each page gets metadata["sections"]: a list of spans {offset, section_no,
    section, section_title, subsection} where offset is a char index into
    page.text. Pages without a heading inherit the section in effect.
    """
    current = {"section_no": None, "section": None, "section_title": None, "subsection": None}
    for page in pages:
        spans = [{"offset": 0, "heading": False, **current}]
        offset = 0
        for line in page.text.split("\n"):
            chapter = _CHAPTER.match(line)
            sub = _SUBSECTION.match(line)
            next_no = (current["section_no"] or 0) + 1
            if chapter and int(chapter.group(1)) == next_no and not line.endswith((".", ",")):
                current = {"section_no": next_no, "section": slugify(chapter.group(2)),
                           "section_title": chapter.group(2).strip(), "subsection": None}
                spans.append({"offset": offset, "heading": True, **current})
            elif sub and current["section_no"] and sub.group(1).split(".")[0] == str(current["section_no"]):
                current = {**current, "subsection": line.strip()}
                spans.append({"offset": offset, "heading": True, **current})
            offset += len(line) + 1
        # Drop a span superseded at the same offset (heading on the first line).
        dedup: list[dict] = []
        for span in spans:
            if dedup and dedup[-1]["offset"] == span["offset"]:
                dedup[-1] = span
            else:
                dedup.append(span)
        page.metadata["sections"] = dedup


def section_at(page: DocumentPage, offset: int) -> dict:
    """Section span in effect at a char offset of page.text."""
    spans = page.metadata.get("sections") or [{"offset": 0}]
    found = spans[0]
    for span in spans:
        if span["offset"] <= offset:
            found = span
        else:
            break
    return {k: v for k, v in found.items() if k not in ("offset", "heading")}


def heading_offsets(page: DocumentPage) -> set[int]:
    """Char offsets (in page.text) of heading lines and of appended OCR block titles."""
    return ({s["offset"] for s in page.metadata.get("sections", []) if s.get("heading")}
            | set(page.metadata.get("block_starts", [])))


def append_ocr_blocks(page: DocumentPage, blocks: list[dict]) -> None:
    """Append accepted OCR text of the page's images after the extracted text.

    Must run after annotate_sections so OCR'd numbers are never read as
    headings; OCR blocks inherit the section in effect at the end of the page.
    Each block starts with a title line "[OCR ảnh k trang p ...]".
    """
    if not blocks:
        return
    starts = []
    text = page.text
    page.metadata["ocr_offset"] = len(text) + (1 if text else 0)
    for k, block in enumerate(blocks, start=1):
        if text:
            text += "\n"
        starts.append(len(text))
        text += (f"[OCR ảnh {k} trang {page.page} — Tesseract, độ tin cậy "
                 f"{block['mean_conf']:.0f}]\n{block['text']}")
    page.text = text
    page.metadata["block_starts"] = starts
    page.metadata["ocr_images"] = len(blocks)
    page.metadata["ocr_chars"] = sum(len(b["text"]) for b in blocks)
