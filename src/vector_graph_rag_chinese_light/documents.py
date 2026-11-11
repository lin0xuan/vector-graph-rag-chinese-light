"""Extract source-located blocks and split them into non-overlapping passages."""
import re
from pathlib import Path

MAX_PASSAGE_LENGTH = 300


def chunks(text):
    text = text.strip()
    while text:
        end = min(MAX_PASSAGE_LENGTH, len(text))
        if end < len(text):
            end = next((i + 1 for i in range(end - 1, MAX_PASSAGE_LENGTH // 2 - 1, -1)
                        if text[i] in "。！？!?；;\n"), end)
        yield text[:end].strip()
        text = text[end:].strip()


def text_blocks(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = 1
    for block in re.split(r"\n[ \t]*\n", text):
        leading = len(block) - len(block.lstrip("\n"))
        if block.strip():
            yield f"第 {start + leading} 行", block
        start += block.count("\n") + 2


def file_blocks(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        yield from text_blocks(path.read_text(encoding="utf-8-sig"))
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF support requires pip install '.[documents]'") from exc
        for number, page in enumerate(PdfReader(path).pages, 1):
            yield f"第 {number} 页", page.extract_text() or ""
    elif suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise ValueError("DOCX support requires pip install '.[documents]'") from exc
        for number, paragraph in enumerate(Document(path).paragraphs, 1):
            yield f"第 {number} 段", paragraph.text
    else:
        raise ValueError(f"Unsupported document format: {suffix}")
