# Copilot instructions for ragonrag

This project is a small Retrieval-Augmented Generation (RAG) demo focused on parsing PDFs and building a knowledge base from research papers. Keep instructions concise and specific to the repository structure so an AI coding agent can be immediately productive.

Key points (be brief):
- Primary purpose: parse research PDFs (under `data/raw/papers/`) and prepare them for downstream RAG experiments. The main code artifact is `code/parse_pdfs.ipynb`.
- There is no build system or tests in the repo. Treat the Jupyter notebook as the runnable entry point for data processing.

What to change and why:
- Prefer edits that keep the notebook runnable in a standard Python environment (Python 3.10+). If adding scripts, add a small `requirements.txt` and a short `README` entry describing how to run the script from the command line.
- Avoid large refactors that introduce many new files or dependencies; this repo is intended to be minimal and educational.

Repository layout to reference in code suggestions:
- `code/parse_pdfs.ipynb` — primary processing notebook. Use this when adding or changing parsing logic.
- `data/raw/papers/` — sample PDF corpus. Use concrete filenames from this folder in examples (e.g., `NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf`).

Conventions and patterns discovered in repo (explicit, discoverable):
- Keep changes minimal and self-contained in `code/` unless adding infra (requirements, scripts). If you add a script counterpart to a notebook, place it under `code/` and include a short CLI (argparse) wrapper.
- No explicit style or formatting configuration detected. Follow sensible defaults: small, well-documented functions, and clear cells in notebooks.

Developer workflows (explicit/safe suggestions):
- To run interactive parsing work, open `code/parse_pdfs.ipynb` in Jupyter or VS Code and run cells top-to-bottom.
- If you convert notebook logic into a script, provide a `requirements.txt` with pinned versions and a short one-line run example in the repo `README.md`.

Integration points & external dependencies:
- PDFs are the only external input and live in `data/raw/papers/`. When adding parsers, prefer well-known Python libraries (e.g., `pdfminer.six`, `pypdf`/`PyPDF2`, or `pdfplumber`) and document any new dependency in `requirements.txt`.

Examples to reference when editing or adding code:
- Use `data/raw/papers/NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf` in unit-test-like examples or small smoke tests.
- When modifying parsing logic, add a short notebook cell or a small script that runs on a single PDF and prints token counts and top extracted headings/sections.

What not to do:
- Don't add heavy infra (Docker, CI) without confirming intent — this repo is a minimal educational demo. Avoid changing global project layout.
- Don't assume presence of hidden config files or secret keys; none were discovered.

If you need to extend the project with durable code (scripts, small modules):
- Add `code/parse_pdfs.py` alongside the notebook with the same core logic and a small CLI:
  - Inputs: path to PDF or folder
  - Outputs: JSONL or text chunks suitable for embeddings
- Add `requirements.txt` and a one-line run command in `README.md`.

When in doubt, ask the repo owner which direction they prefer before adding CI or major refactors.

Quick checklist for PRs from an AI agent:
1. Keep changes small and documented in `README.md` or notebook cells.
2. If new dependencies are added, include a `requirements.txt` and a short reasoning comment in PR description.
3. Include a minimal smoke-test cell/script demonstrating the change on one PDF.

Feedback request:
Please review this guidance and tell me if you want broader instructions (CI, packaging, tests) or prefer to keep it minimal and notebook-focused.
