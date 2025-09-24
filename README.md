# ragonrag
This is a RAG application to answer questions about building RAG applications. 

To create JSONL chunks from the sample PDFs and/or a list of blog URLs:
    
    # parse all PDFs in the sample folder:
    python code/parse_and_chunk.py --input data/raw/papers/ --out data/processed/papers_chunks.jsonl

    # parse a single PDF:
    python code/parse_and_chunk.py --input data/raw/papers/2020.emnlp-main.550.pdf --out data/processed/example_pdf.jsonl

    # scrape URLs from a file urls.txt (one URL per line):
    python code/parse_and_chunk.py --input urls.txt --out data/processed/blog_chunks.jsonl

## Google Document AI prototype

If you'd like to use Google Document AI (managed, high-quality layout parsing) to extract structured blocks and reading order from PDFs, there's a small prototype script at `src/docai_prototype.py`.

Quick steps:

1. Enable the Document AI API in the Google Cloud Console and create a Processor (e.g. 'Document OCR' / 'Form Parser').
2. Create a service account with the Document AI role and download the JSON key. Set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

3. Run the prototype (synchronous, suitable for small PDFs):

```bash
python -m src.docai_prototype \
    --processor-name projects/PROJECT_ID/locations/LOCATION/processors/PROCESSOR_ID \
    --input data/raw/papers/NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf \
    --out data/processed/docai_example.jsonl
```

The script writes a JSONL file where each line is a block with `page`, `bbox` (normalized vertices), and `text`. For larger jobs or many PDFs, prefer the async/batch Document AI APIs.