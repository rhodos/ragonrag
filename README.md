# ragonrag
This is a RAG application to answer questions about building RAG applications. 

To create JSONL chunks from the sample PDFs and/or a list of blog URLs:
    
    # parse all PDFs in the sample folder:
    python code/parse_and_chunk.py --input data/raw/papers/ --out data/processed/papers_chunks.jsonl

    # parse a single PDF:
    python code/parse_and_chunk.py --input data/raw/papers/2020.emnlp-main.550.pdf --out data/processed/example_pdf.jsonl

    # scrape URLs from a file urls.txt (one URL per line):
    python code/parse_and_chunk.py --input urls.txt --out data/processed/blog_chunks.jsonl