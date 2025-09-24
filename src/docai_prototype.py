"""Small prototype for using Google Document AI to parse PDFs and output structured JSONL.

This script is intentionally minimal and synchronous for small PDFs. It expects the
environment variable GOOGLE_APPLICATION_CREDENTIALS to point to a service account JSON key
with permission to call the Document AI processor you create in your Google Cloud project.

Usage (example):
  python -m src.docai_prototype \
    --processor-name projects/PROJECT_ID/locations/LOCATION/processors/PROCESSOR_ID \
    --input data/raw/papers/NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf \
    --out data/processed/docai_example.jsonl

Notes:
- For large or many PDFs, use the async/batch API. This script uses the synchronous `process_document`
  method which is fine for small PDFs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from google.cloud import documentai_v1 as documentai

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class Block:
    page: int
    bbox: dict
    text: str


def get_text_for_anchor(document: documentai.Document, text_anchor) -> str:
    if not text_anchor or not getattr(text_anchor, "text_segments", None):
        return ""
    out = []
    for seg in text_anchor.text_segments:
        start_idx = int(seg.start_index or 0)
        end_idx = int(seg.end_index)
        out.append(document.text[start_idx:end_idx])
    return "".join(out)


def process_pdf_sync(processor_name: str, path: Path) -> List[Block]:
    """Process a PDF via Document AI sync API and return ordered blocks.

    Returns a list of Block(page, bbox, text). Bbox is normalized vertices list.
    """
    client = documentai.DocumentProcessorServiceClient()
    with open(path, "rb") as f:
        pdf_bytes = f.read()

    raw_doc = {"content": pdf_bytes, "mime_type": "application/pdf"}
    request = {"name": processor_name, "raw_document": raw_doc}

    logger.info("Sending %s to Document AI processor %s", path, processor_name)
    result = client.process_document(request=request)
    doc = result.document

    blocks: List[Block] = []
    for p in doc.pages:
        page_num = int(p.page_number)
        # Use blocks and paragraphs where available, fallback to lines
        candidates = list(p.blocks or [])
        if not candidates:
            candidates = list(p.paragraphs or [])
        if not candidates:
            candidates = list(p.lines or [])

        for b in candidates:
            text = get_text_for_anchor(doc, b.layout.text_anchor)
            bbox = [v.to_dict() for v in b.layout.bounding_poly.normalized_vertices]
            blocks.append(Block(page=page_num, bbox=bbox, text=text))

    # Optionally sort by page, top->left using bbox centroid
    def centroid_yx(bb):
        ys = [v.get("y", 0) for v in bb]
        xs = [v.get("x", 0) for v in bb]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    blocks.sort(key=lambda b: (b.page, *centroid_yx(b.bbox)))
    return blocks


def blocks_to_jsonl(blocks: List[Block], out_path: Path):
    with out_path.open("w", encoding="utf-8") as fh:
        for b in blocks:
            json.dump({"page": b.page, "bbox": b.bbox, "text": b.text}, fh, ensure_ascii=False)
            fh.write("\n")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--processor-name", required=True, help="Document AI processor resource name")
    p.add_argument("--input", required=True, help="Path to input PDF")
    p.add_argument("--out", required=True, help="Output JSONL path")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    if not inp.exists():
        logger.error("Input not found: %s", inp)
        raise SystemExit(2)

    blocks = process_pdf_sync(args.processor_name, inp)
    blocks_to_jsonl(blocks, out)
    logger.info("Wrote %d blocks to %s", len(blocks), out)


if __name__ == "__main__":
    main()
