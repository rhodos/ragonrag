import argparse
import json
import pandas as pd
from pathlib import Path
import uuid
from typing import List, Dict

import pdfplumber
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

# Configure module-level logger
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "parse_and_chunk.log"

logger = logging.getLogger("parse_and_chunk")
logger.setLevel(logging.DEBUG)

# Avoid adding duplicate handlers if this module is reloaded
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

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

def fetch_url(url: str, max_retries: int = 3) -> requests.Response:
    # Try with a browser-like User-Agent. If we get a 403, retry once with
    # an alternate UA and a Referer header to improve chances.
    default_ua = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    fallback_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )

    attempts = [
        {"User-Agent": default_ua},
        {"User-Agent": fallback_ua, "Referer": url},
    ]

    resp = None
    for i, hdrs in enumerate(attempts, start=1):
        try:
            logger.debug(f"Requesting {url} (attempt {i}) with UA={hdrs.get('User-Agent')}")
            resp = requests.get(url, timeout=15, headers=hdrs)
            if resp.status_code == 403:
                logger.warning(f"Received 403 for {url} on attempt {i} with UA={hdrs.get('User-Agent')}")
                resp = None
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            logger.warning(f"Request attempt {i} for {url} failed: {e}")
            resp = None
            continue

    if resp is None:
        # let the exception bubble up with a clear message
        raise requests.HTTPError(f"Failed to fetch {url} after {len(attempts)} attempts (likely 403 or network error)")
    
    return resp

# TODO: Improve scraping to remove ads, include numbers in numbered lists, and either annotate figure captions as such, or remove them. 
def scrape_url(url: str) -> str:
    resp = fetch_url(url)

    soup = BeautifulSoup(resp.text, "html.parser")

    # remove noisy elements
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        tag.decompose()

    # prefer <article> if available
    article = None
    for cls in ["article", "main-content", "post-content", "article-content", "content"]:
        article = soup.find(class_=cls)
        if article:
            logger.info(f"Found article by class '{cls}' for {url}")
            break
    if not article:
        logger.warning(f"No article tag found for {url}, using full page.")
    base = article if article else soup

    # preserve code blocks (pre/code) by marking them
    for code in base.find_all(["pre", "code"]):
        logger.debug(f"Inferred code block in {url}: {code.get_text()[:30]}...")
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

def process_url(url: str, max_words: int, overlap: int, metadata: dict) -> List[Dict]:
    text = scrape_url(url)
    chunks = chunk_text(text, max_words=max_words, overlap=overlap)
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "id": str(uuid.uuid4()),
            "source": url,
            "chunk_index": i,
            "text": c,
            "metadata": metadata
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
            try:
                items.extend(process_pdf_file(pdf, args.max_words, args.overlap))
            except Exception as e:
                print(f"Warning: failed to process {pdf}: {e}")
    elif inp.is_file() and inp.suffix.lower() == ".pdf":
        items.extend(process_pdf_file(inp, args.max_words, args.overlap))
    elif inp.is_file() and inp.suffix.lower() == ".tsv":
        file_data = pd.read_csv(inp, sep="\t")
        #with inp.open() as f:
        #    urls = [ln.strip() for ln in f if ln.strip()]
        for i, row in tqdm(file_data.iterrows(), desc="articles"):
            url = row['url']  # assuming the TSV has a column named 'url'
            try:
                items.extend(process_url(url, args.max_words, args.overlap, metadata=row.to_dict()))
            except Exception as e:
                print(f"Warning: failed to scrape {url}: {e}")
    else:
        raise SystemExit("Input must be a pdf, a folder, or a tsv file with URLs and metadata")

    write_jsonl(items, Path(args.out))
    print(f"Wrote {len(items)} chunks to {args.out}")

if __name__ == "__main__":
    main()