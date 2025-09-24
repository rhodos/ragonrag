"""Smoke test for the Google Document AI prototype.

This test is intentionally guarded: it will be skipped unless the environment is configured with `GOOGLE_APPLICATION_CREDENTIALS` and the `google-cloud-documentai` client is installed.

Run locally after configuring the service account and creating a Document AI processor.
"""
import os
import tempfile
from pathlib import Path
import logging
from src import docai_prototype

import pytest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


def has_docai_client():
    try:
        import google.cloud.documentai_v1 as _
        logger.debug("google-cloud-documentai is installed")
        return True
    except Exception:
        logger.debug("google-cloud-documentai is NOT installed")
        return False
    
def google_credentials_set():
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds:
        logger.debug("GOOGLE_APPLICATION_CREDENTIALS is set")
        return True
    else:
        logger.debug("GOOGLE_APPLICATION_CREDENTIALS is NOT set")
        return False

@pytest.mark.skipif(
    not google_credentials_set() or not has_docai_client(),
    reason="Document AI not configured or client library not installed",
)
def test_docai_smoke_runs_and_writes_jsonl():


    # Use a small, representative PDF from the repo
    pdf = Path("data/raw/papers/NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf")
    assert pdf.exists(), "sample PDF not found in repo"

    processor = os.environ.get("DOCAI_PROCESSOR_NAME")
    assert processor, "Set DOCAI_PROCESSOR_NAME env var to 'projects/.../locations/.../processors/...'"

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "docai_out.jsonl"
        blocks = docai_prototype.process_pdf_sync(processor, pdf)

        # Basic sanity checks
        assert isinstance(blocks, list)
        assert len(blocks) > 0, "Expected at least one block from Document AI"

        docai_prototype.blocks_to_jsonl(blocks, out)
        assert out.exists()
        # File should be non-empty
        assert out.stat().st_size > 0
