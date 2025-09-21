import argparse
import json
import os
from pathlib import Path
import uuid
from typing import List, Dict

import pdfplumber
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Simple chunking: split by words into approximately `max_words` chunks with `overlap` words
def chunk_text(text: str, max_words: int = 250, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap
    return chunks

def parse_pdf(path: Path) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            texts.append(text)
    return "\n\n".join(texts).strip()

def scrape_url(url: str) -> str:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # remove noisy elements
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        tag.decompose()
    
    # prefer <article> if available
    article = soup.find("article")
    base = article if article else soup
    
    # preserve code blocks (pre/code) by marking them
    for code in base.find_all(["pre", "code"]):
        code.insert_before("\n\n[CODEBLOCK]\n")
        code.insert_after("\n\n[/CODEBLOCK]\n")
    text = base.get_text(separator="\n")
    
    # collapse whitespace
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text

def process_pdf_file(path: Path, max_words: int, overlap: int) -> List[Dict]:
    text = parse_pdf(path)
    chunks = chunk_text(text, max_words=max_words, overlap=overlap)
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "id": str(uuid.uuid4()),
            "source": str(path),
            "chunk_index": i,
            "text": c,
            "metadata": {
                "filename": path.name,
                "type": "pdf",
            }
        })
    return out

def process_url(url: str, max_words: int, overlap: int) -> List[Dict]:
    text = scrape_url(url)
    chunks = chunk_text(text, max_words=max_words, overlap=overlap)
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "id": str(uuid.uuid4()),
            "source": url,
            "chunk_index": i,
            "text": c,
            "metadata": {
                "url": url,
                "type": "web",
            }
        })
    return out

def write_jsonl(items: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    p = argparse.ArgumentParser(description="Parse PDFs and scrape URLs into JSONL chunks")
    p.add_argument("--input", required=True, help="PDF file, folder of PDFs, or text file of URLs")
    p.add_argument("--out", required=True, help="Output JSONL file")
    p.add_argument("--max-words", type=int, default=250, help="Approx words per chunk")
    p.add_argument("--overlap", type=int, default=50, help="Word overlap between chunks")
    args = p.parse_args()

    inp = Path(args.input)
    items = []

    if inp.is_dir():
        pdfs = sorted(inp.glob("*.pdf"))
        for pdf in tqdm(pdfs, desc="PDFs"):
            items.extend(process_pdf_file(pdf, args.max_words, args.overlap))
    elif inp.is_file() and inp.suffix.lower() == ".pdf":
        items.extend(process_pdf_file(inp, args.max_words, args.overlap))
    elif inp.is_file():
        # treat as file containing newline-separated URLs
        with inp.open() as f:
            urls = [ln.strip() for ln in f if ln.strip()]
        for url in tqdm(urls, desc="URLs"):
            try:
                items.extend(process_url(url, args.max_words, args.overlap))
            except Exception as e:
                print(f"Warning: failed to scrape {url}: {e}")
    else:
        raise SystemExit("Input must be a pdf, a folder, or a file with URLs")

    write_jsonl(items, Path(args.out))
    print(f"Wrote {len(items)} chunks to {args.out}")

if __name__ == "__main__":
    main()