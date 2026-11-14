"""OCR for images embedded in PDF pages (separate step from text extraction).

Engine: Tesseract 5 with Vietnamese data, run in Docker (image built from
starter/ocr/Dockerfile) so nothing has to be installed on the host.

For every content image (images repeated on >=60% of pages, e.g. the header
logo, are skipped) two variants are OCR'd with --psm 4:
  gray  - grayscale, upscaled 2x when narrower than 1600 px
  bin   - gray + autocontrast + threshold 150 (helps coloured table cells)
The variant with the higher mean word confidence is kept. Results whose mean
confidence or amount of clean text is too low (photos, line charts) are
recorded but not added to the corpus.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps
from pypdf import PdfReader

from .ingestion import normalize_text

DOCKER_IMAGE = "day6-ocr:tesseract-vie"
TESSERACT_ARGS = ["-l", "vie", "--psm", "4"]
MIN_CONFIDENCE = 80.0  # label photos (p9-10) score 71-78 and are mostly noise; tables >= 85
MIN_CLEAN_CHARS = 40


def _clean_lines(text: str) -> str:
    """Drop OCR noise lines: too short or mostly non-alphanumeric."""
    kept = []
    for line in normalize_text(text).split("\n"):
        line = line.strip(" |_—-")
        if len(line) < 3:
            continue
        alnum = sum(ch.isalnum() for ch in line)
        if alnum / len(line) < 0.5:
            continue
        kept.append(line)
    return "\n".join(kept)


def _mean_conf(tsv_path: Path) -> tuple[float, int]:
    confs = []
    with tsv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            try:
                conf = float(row.get("conf", -1))
            except ValueError:
                continue
            if conf >= 0 and (row.get("text") or "").strip():
                confs.append(conf)
    return (sum(confs) / len(confs) if confs else 0.0), len(confs)


def extract_images(pdf_path: Path, out_dir: Path, min_share: float = 0.6) -> list[dict]:
    """Save content images and their two preprocessed variants."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    found = []
    for page_no, page in enumerate(reader.pages, start=1):
        for idx, img in enumerate(page.images):
            data = img.data
            found.append((page_no, idx, img.name, hashlib.sha1(data).hexdigest(), data))
    pages_per_hash = Counter()
    for h in {(p, h) for p, _, _, h, _ in found}:
        pages_per_hash[h[1]] += 1
    threshold = max(2, int(len(reader.pages) * min_share))
    images = []
    for page_no, idx, name, digest, data in found:
        if pages_per_hash[digest] >= threshold:
            continue  # running logo / decoration
        im = Image.open(io.BytesIO(data)).convert("L")
        stem = f"p{page_no:03d}_{idx}"
        gray = im.resize((im.width * 2, im.height * 2), Image.LANCZOS) if im.width < 1600 else im
        binary = ImageOps.autocontrast(gray).point(lambda v: 255 if v > 150 else 0)
        gray.save(out_dir / f"{stem}_gray.png")
        binary.save(out_dir / f"{stem}_bin.png")
        images.append({"page": page_no, "image_index": idx, "pdf_name": name,
                       "sha1": digest, "size": [im.width, im.height], "stem": stem})
    return images


def run_tesseract(work_dir: Path, stems: list[str]) -> None:
    """OCR all variants in one container; writes <file>.txt and <file>.tsv."""
    script = " ; ".join(
        f"tesseract {stem}_{v}.png {stem}_{v} {' '.join(TESSERACT_ARGS)} txt tsv 2>/dev/null"
        for stem in stems for v in ("gray", "bin"))
    cmd = ["docker", "run", "--rm", "-v", f"{work_dir.resolve()}:/data", "-w", "/data",
           "--entrypoint", "sh", DOCKER_IMAGE, "-c", script]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Tesseract container failed (build it with "
                           f"`docker build -t {DOCKER_IMAGE} ocr`): {proc.stderr[-500:]}")


def tesseract_version() -> str:
    proc = subprocess.run(["docker", "run", "--rm", DOCKER_IMAGE, "--version"],
                          capture_output=True, text=True, encoding="utf-8")
    return (proc.stdout or proc.stderr).splitlines()[0] if (proc.stdout or proc.stderr) else "?"


def ocr_pdf(pdf_path: Path, document_id: str, work_dir: Path) -> dict:
    images = extract_images(pdf_path, work_dir)
    run_tesseract(work_dir, [i["stem"] for i in images])
    results = []
    for info in images:
        variants = {}
        for v in ("gray", "bin"):
            conf, words = _mean_conf(work_dir / f"{info['stem']}_{v}.tsv")
            raw = (work_dir / f"{info['stem']}_{v}.txt").read_text(encoding="utf-8")
            variants[v] = {"mean_conf": round(conf, 1), "words": words, "text": _clean_lines(raw)}
        best = max(variants, key=lambda v: variants[v]["mean_conf"])
        chosen = variants[best]
        accepted = chosen["mean_conf"] >= MIN_CONFIDENCE and len(chosen["text"]) >= MIN_CLEAN_CHARS
        reason = "ok" if accepted else (
            f"mean_conf {chosen['mean_conf']} < {MIN_CONFIDENCE}"
            if chosen["mean_conf"] < MIN_CONFIDENCE else
            f"clean text {len(chosen['text'])} < {MIN_CLEAN_CHARS} chars")
        results.append({
            **{k: info[k] for k in ("page", "image_index", "pdf_name", "sha1", "size")},
            "document_id": document_id, "variant": best,
            "mean_conf": chosen["mean_conf"], "words": chosen["words"],
            "conf_by_variant": {v: variants[v]["mean_conf"] for v in variants},
            "chars": len(chosen["text"]), "accepted": accepted, "reason": reason,
            "text": chosen["text"],
        })
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "engine": tesseract_version(), "docker_image": DOCKER_IMAGE,
        "args": TESSERACT_ARGS, "min_confidence": MIN_CONFIDENCE,
        "min_clean_chars": MIN_CLEAN_CHARS, "images": results,
    }


def load_ocr(path: Path) -> dict[tuple[str, int], list[dict]]:
    """Accepted OCR blocks grouped by (document_id, page), in image order."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], list[dict]] = {}
    for item in data["images"]:
        if item["accepted"]:
            out.setdefault((item["document_id"], item["page"]), []).append(item)
    return out


OCR_MARKER = re.compile(r"^\[OCR ảnh \d+ trang \d+")
