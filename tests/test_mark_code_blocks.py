import pytest
from bs4 import BeautifulSoup

from src.document_parser import mark_code_blocks


def test_mark_code_blocks_marks_multiline_only():
    html = """
    <html>
      <body>
        <p>Here is some inline code: <code>inline_func()</code> in a sentence.</p>
        <div class="code">single line code()</div>
        <pre><code>def foo():
    print('line1')
    print('line2')</code></pre>
      </body>
    </html>
    """

    soup = BeautifulSoup(html, "html.parser")
    # call the function under test
    mark_code_blocks(soup, url="http://example.test")

    text = soup.get_text()

    # Inline code and single-line .code should be present
    inline_idx = text.find("inline_func()")
    single_idx = text.find("single line code()")
    assert inline_idx != -1
    assert single_idx != -1

    # Multi-line code block should be wrapped by markers
    start = text.find("[CODEBLOCK]")
    end = text.find("[/CODEBLOCK]")
    assert start != -1 and end != -1 and end > start
    assert "print('line1')" in text[start:end]

    # Ensure inline and single-line code are NOT inside the CODEBLOCK region
    assert not (start < inline_idx < end), "inline code was incorrectly wrapped"
    assert not (start < single_idx < end), "single-line code was incorrectly wrapped"
