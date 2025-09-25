"""Small prototype for using Google Document AI to parse PDFs and output structured JSONL.

This script is intentionally minimal and synchronous for small PDFs. It expects the environment variable GOOGLE_APPLICATION_CREDENTIALS to point to a service account JSON key with permission to call the Document AI processor you create in your Google Cloud project. It also expects the environment variable DOCAI_PROCESSOR_NAME to be set to the full resource name of your processor, e.g.:
  projects/PROJECT_ID/locations/LOCATION/processors/PROCESSOR_ID

Usage (example):
  python -m src.docai_prototype \
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
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import os
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import pandas as pd
import io
from pypdf import PdfReader, PdfWriter

from google.cloud import documentai_v1 as documentai

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

load_dotenv()

@dataclass
class Block:
    page: int
    bbox: List[Dict[str, float]]
    text: str
    doc: str
    metadata: Optional[Dict[str, Any]] = None


def get_text_for_anchor(document: documentai.Document, text_anchor) -> str:
    if not text_anchor or not getattr(text_anchor, "text_segments", None):
        return ""
    out = []
    for seg in text_anchor.text_segments:
        start_idx = int(seg.start_index or 0)
        end_idx = int(seg.end_index)
        out.append(document.text[start_idx:end_idx])
    return "".join(out)


def process_small_pdf(client, processor_name: str, pdf_name: str, pdf_bytes, metadata=None, page_offset: int=0) -> Tuple[List[Block], bool]:
    """Process a PDF via Document AI sync API and return ordered blocks.

    Returns a list of Block(page, bbox, text). Bbox is normalized vertices list.
    """
    # with open(pdf_path, "rb") as f:
    #     pdf_bytes = f.read()

    raw_doc = {"content": pdf_bytes, "mime_type": "application/pdf"}
    request = {"name": processor_name, "raw_document": raw_doc}

    logger.info(f"Sending {pdf_name} to Document AI processor {processor_name}")
    result = client.process_document(request=request)
    doc = result.document

    blocks: List[Block] = []
    num_pages = len(doc.pages)
    logger.info(f"Document AI returned {num_pages} pages for {pdf_name}")

    for p in doc.pages:
        # Document AI returns page_number starting at 1 for the returned document. If we
        # processed a chunk of the original PDF, apply a page_offset to map to the
        # original global page numbers.
        local_page_num = int(p.page_number)
        page_num = int(page_offset) + local_page_num
        
        # Use blocks and paragraphs where available, fallback to lines
        candidates = list(p.blocks or [])
        if not candidates:
            logger.warning(f"No blocks found on page {page_num} of {pdf_name}, falling back to paragraphs")
            candidates = list(p.paragraphs or [])
        if not candidates:
            logger.warning(f"No paragraphs found on page {page_num} of {pdf_name}, falling back to lines")
            candidates = list(p.lines or [])

        for b in candidates:
            text = get_text_for_anchor(doc, b.layout.text_anchor)
            verts = getattr(b.layout.bounding_poly, "normalized_vertices", []) or []
            bbox = [_vertex_to_dict(v) for v in verts]
            blocks.append(Block(page=page_num, bbox=bbox, text=text, metadata=metadata, doc=pdf_name))

    # # Sort by page, top->left using bbox centroid
    # def centroid_yx(bb):
    #     ys = [v.get("y", 0) for v in bb]
    #     xs = [v.get("x", 0) for v in bb]
    #     return (sum(ys) / len(ys), sum(xs) / len(xs))

    # blocks.sort(key=lambda b: (b.page, *centroid_yx(b.bbox)))

    (idx, ref_found) = _look_for_references_block(blocks)
    if ref_found:
        logger.info(f"Found References section starting at block index {idx} in document {pdf_name}")
        blocks = blocks[:idx]  # truncate at References
    else:
        logger.info(f"No References section found in document {pdf_name}")

    return (blocks, ref_found)

def _vertex_to_dict(v):
    return {"x": float(v.x), "y": float(v.y)}

def _look_for_references_block(blocks: List[Block]):
    """Heuristic to find the start of the References section in a list of Blocks.

    Returns the index of the block that starts the References section, or None if not found.
    """
    idx = -1
    found = False
    for i, b in enumerate(blocks):
        if b.text.strip().lower() in ("references", "bibliography"):
            idx = i
        if b.text.strip().lower().startswith("references\n") or b.text.strip().lower().startswith("bibliography\n"):
            idx = i
        if "references" in b.text.strip().lower()[:20] or "bibliography" in b.text.strip().lower()[:20]:
            idx = i
    if idx != -1:
        found = True
    return (idx, found)


def process_pdf(processor_name: str, pdf_path: Path, metadata: Optional[Dict[str, Any]]=None, maxpages: int=15) -> List[Block]:

    # If the PDF is large, some Document AI sync endpoints may fail; for robustness,
    # split into chunks of up to `maxpages` pages and call the API per chunk when necessary.

    client = documentai.DocumentProcessorServiceClient()

    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    logger.info(f"PDF {pdf_path} has {page_count} pages")
 
    if page_count <= maxpages:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        blocks, _ = process_small_pdf(client=client, processor_name=processor_name, pdf_name=pdf_path.name, pdf_bytes=pdf_bytes, metadata=metadata)
    elif page_count > maxpages:
        logger.info("Large PDF (%d pages) detected; splitting into chunks of up to %d pages", page_count, maxpages)

        blocks: List[Block] = []
        # iterate chunks and process each as an in-memory PDF
        for start in range(0, page_count, maxpages):
            end = min(start + maxpages, page_count)
            writer = PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)
            buf.seek(0)
            pdf_bytes = buf.read()
            # Pass page_offset=start so returned page numbers are mapped to global pages
            new_blocks, ref_found = process_small_pdf(client=client, processor_name=processor_name, pdf_name=pdf_path.name, pdf_bytes=pdf_bytes, metadata=metadata, page_offset=start)
            blocks.extend(new_blocks)
            if ref_found:
                logger.info(f"Stopping early at chunk starting page {start+1} due to References section")
                break
    
    return blocks

def blocks_to_jsonl(blocks: List[Block], out_path: Path, write_bbox: bool=False):
    with out_path.open("w", encoding="utf-8") as fh:
        for b in blocks:
            if write_bbox:
                json.dump({"page": b.page, "bbox": b.bbox, "text": b.text, "metadata": b.metadata}, fh, ensure_ascii=False)
            else:
                json.dump({"page": b.page, "text": b.text, "metadata": b.metadata}, fh, ensure_ascii=False)
            fh.write("\n")

def get_processor_name():
    processor_name = os.environ.get("DOCAI_PROCESSOR_NAME")
    if not processor_name:
        logger.error("Set DOCAI_PROCESSOR_NAME environment variable to your Document AI processor name")
        raise SystemExit(2)
    return processor_name


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--input", default='data/raw/papers/2020.emnlp-main.550.pdf', help="Path to input PDF, directory containing PDFs, or TSV file of PDF paths and corresponding metadata")
    p.add_argument("--output", default='data/processed/docai_example.jsonl', help="Output JSONL path")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        logger.error("Input not found: %s", inp)
        raise SystemExit(2)
    out = Path(args.output)
    
    processor = get_processor_name()
    
    blocks = []

    if inp.is_dir():
        pdfs = sorted(inp.glob("*.pdf"))
        for pdf in tqdm(pdfs, desc="PDFs"):
            try:
                blocks.extend(process_pdf(processor_name=processor, pdf_path=Path(pdf)))
            except Exception as e:
                print(f"Warning: failed to process {pdf}: {e}")
    elif inp.is_file() and inp.suffix.lower() == ".pdf":
        blocks.extend(process_pdf(processor_name=processor, pdf_path=inp))
    elif inp.is_file() and inp.suffix.lower() == ".tsv":
        file_data = pd.read_csv(inp, sep="\t")
        for _, row in tqdm(file_data.iterrows(), desc="PDFs"):
            pdf = row['pdf_path']  # assuming the TSV has a column named 'pdf_path'
            try:
                blocks.extend(process_pdf(processor_name=processor, pdf_path=Path(pdf), metadata=row.to_dict()))
            except Exception as e:
                print(f"Warning: failed to process pdf {pdf}: {e}")
    else:
        raise SystemExit("Input must be a PDF, a folder, or a tsv file with PDFs and metadata")

    blocks_to_jsonl(blocks, out, write_bbox=False)
    logger.info("Wrote %d blocks to %s", len(blocks), out)


if __name__ == "__main__":
    main()
