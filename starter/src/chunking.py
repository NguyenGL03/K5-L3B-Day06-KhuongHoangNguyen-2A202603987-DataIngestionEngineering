"""Fixed-size and structure-aware chunking. Chunks never cross a page boundary,
so every chunk keeps an exact (document_id, source, page) trace."""
from __future__ import annotations

import re

from .ingestion import heading_offsets, section_at
from .models import Chunk, DocumentPage

_BULLET = re.compile(r"^[–•∗\-+*]\s?")
_CAPTION = re.compile(r"^Hình \d+:")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
# pdfTeX wraps body text at ~95-105 chars; a shorter line usually ends a paragraph.
WRAP_WIDTH = 80


def _base_metadata(page: DocumentPage, start: int, end: int, index: int, strategy: str,
                   section_offset: int | None = None) -> dict:
    ocr_offset = page.metadata.get("ocr_offset")
    if ocr_offset is None or end <= ocr_offset:
        content_source = "pdf_text"
    elif start >= ocr_offset:
        content_source = "ocr"
    else:
        content_source = "mixed"
    return {
        **section_at(page, start if section_offset is None else section_offset),
        "content_source": content_source,
        "printed_page": page.metadata.get("printed_page"),
        "char_start": start,
        "char_end": end,
        "chunk_index": index,
        "chunker": strategy,
    }


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split page text into overlapping character windows.

    The window end is moved back to the nearest whitespace (at most 30 chars)
    so a word is not cut in half; otherwise structure is ignored on purpose:
    this is the baseline. Section metadata is the section in effect at the
    first character of the window.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = page.text
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", end - 30, end)
            cut = max(cut, text.rfind("\n", end - 30, end))
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            i = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{page.document_id}-p{page.page:03d}-fixed{size}o{overlap}-{i:03d}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=piece,
                metadata=_base_metadata(page, start, end, i, f"fixed_{size}_{overlap}"),
            ))
        if end >= len(text):
            break
        next_start = end - overlap
        # Start the overlap on a word boundary as well.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def _paragraphs(page: DocumentPage) -> list[tuple[int, int, str, bool]]:
    """Rebuild paragraphs from hard-wrapped PDF lines.

    Returns (start, end, text, is_heading). A new paragraph starts at a heading,
    a list bullet, a figure caption, after a line shorter than WRAP_WIDTH, or
    after a line ending with sentence punctuation. Over-splitting is harmless
    because by_paragraph regroups paragraphs up to max_size; wrongly merging
    two paragraphs is not. Wrapped lines are joined with a space.
    """
    headings = heading_offsets(page)
    paras: list[tuple[int, int, str, bool]] = []
    current: list[str] = []
    cur_start = 0
    prev_len = 0
    prev_end_punct = False
    offset = 0

    def flush(end: int) -> None:
        if current:
            paras.append((cur_start, end, " ".join(current), False))
            current.clear()

    for line in page.text.split("\n"):
        line_end = offset + len(line)
        if offset in headings:
            flush(offset - 1)
            paras.append((offset, line_end, line, True))
            prev_len = 0
        elif not line:
            flush(offset - 1)
            prev_len = 0
        else:
            starts_new = (not current or _BULLET.match(line) or _CAPTION.match(line)
                          or prev_len < WRAP_WIDTH or prev_end_punct)
            if starts_new:
                flush(offset - 1)
                cur_start = offset
            current.append(line)
            prev_len = len(line)
            prev_end_punct = line.endswith((".", ":", ";", "!", "?"))
        offset = line_end + 1
    flush(len(page.text))
    return paras


def _split_long(text: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at sentence ends, then at whitespace."""
    pieces: list[str] = []
    buf = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_size:
            cut = sentence.rfind(" ", 0, max_size)
            cut = cut if cut > 0 else max_size
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if buf and len(buf) + 1 + len(sentence) > max_size:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group consecutive paragraphs into chunks of at most max_size chars.

    A heading always starts a new chunk and stays attached to the text after it,
    so a chunk never mixes two (sub)sections. Paragraphs longer than max_size
    are split at sentence boundaries. A chunk that opens with headings may
    exceed max_size by the length of those heading lines (never split a
    heading from its first paragraph).
    """
    groups: list[tuple[int, int, list[str], int]] = []
    cur: list[str] = []
    cur_start = cur_end = meta_offset = 0
    only_headings = False  # True while `cur` holds nothing but heading lines

    def flush() -> None:
        nonlocal cur
        if cur:
            groups.append((cur_start, cur_end, cur, meta_offset))
            cur = []

    for start, end, text, is_heading in _paragraphs(page):
        if is_heading and not only_headings:
            flush()
        parts = [text] if len(text) <= max_size else _split_long(text, max_size)
        for part in parts:
            size = sum(len(t) + 1 for t in cur) + len(part)
            if cur and size > max_size and not only_headings:
                flush()
            if not cur:
                cur_start = meta_offset = start
            if is_heading:
                meta_offset = start  # section of the last heading opening the chunk
            cur.append(part)
            cur_end = end
            only_headings = is_heading and (only_headings or len(cur) == 1)
    flush()

    chunks: list[Chunk] = []
    for i, (start, end, parts, meta_offset) in enumerate(groups):
        chunks.append(Chunk(
            chunk_id=f"{page.document_id}-p{page.page:03d}-para{max_size}-{i:03d}",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text="\n".join(parts),
            metadata=_base_metadata(page, start, end, i, f"paragraph_{max_size}", meta_offset),
        ))
    return chunks
