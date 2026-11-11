"""Reconstruct PDF blocks from coordinates, with per-page OCR for sparse text layers."""

import csv
import logging
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from contextlib import closing
from ctypes import byref, c_uint
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from .documents import SENTENCE_END, BlockBuilder, Box, normalize, table_markdown

LOGGER = logging.getLogger(__name__)
MIN_TEXT_CHARS = 50


@dataclass(frozen=True)
class Line:
    text: str
    box: Box
    size: float = 0
    bold: bool = False
    kind: str = "paragraph"
    level: int | None = None


def bounds(items):
    return (min(item[0] for item in items), min(item[1] for item in items),
            max(item[2] for item in items), max(item[3] for item in items))


def word_box(word):
    return (word["x0"], word["top"], word["x1"], word["bottom"])


def text_rows(words, tolerance=3):
    rows = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not rows or word["top"] - rows[-1][0]["top"] > tolerance:
            rows.append([])
        rows[-1].append(word)
    return [sorted(row, key=lambda item: item["x0"]) for row in rows]


def word_groups(row, gap):
    groups = []
    for word in row:
        if not groups or word["x0"] - groups[-1][-1]["x1"] >= gap:
            groups.append([])
        groups[-1].append(word)
    return groups


def word_line(words):
    return Line(" ".join(word["text"] for word in words), bounds([word_box(word) for word in words]),
                median(word.get("size", 0) for word in words),
                any(re.search(r"bold|heavy|black|demi", word.get("fontname", ""), re.I) for word in words))


def pale(char):
    color = char.get("non_stroking_color")
    if not isinstance(color, (tuple, list)) or len(color) not in {1, 3, 4}:
        return False
    return all(value < 0.20 for value in color) if len(color) == 4 else all(value > 0.80 for value in color)


def aligned_tables(rows):
    """Recognize consecutive rows with three or more consistently aligned text cells."""
    run = []
    for groups in [*rows, []]:
        aligned = len(groups) >= 3 and (not run or (
            len(groups) == len(run[-1])
            and all(abs(a[0]["x0"] - b[0]["x0"]) <= 6 for a, b in zip(groups, run[-1]))
            and groups[0][0]["top"] - run[-1][0][0]["bottom"] <= 25
        ))
        if not aligned and run:
            if len(run) >= 3:
                cells = [[word_line(group) for group in row] for row in run]
                yield Line(table_markdown([[cell.text for cell in row] for row in cells]),
                           bounds([cell.box for row in cells for cell in row]), kind="table")
            run = []
        if len(groups) >= 3:
            run.append(groups)


def inside(word, box):
    x0, y0, x1, y1 = word_box(word)
    return box[0] <= (x0 + x1) / 2 <= box[2] and box[1] <= (y0 + y1) / 2 <= box[3]


def transparent_text(pdf, page):
    from pypdfium2 import raw

    with closing(pdf[page.page_number - 1]) as image_page:
        if page.rotation:
            return []
        boxes = []
        for obj in image_page.get_objects(filter=[raw.FPDF_PAGEOBJ_TEXT]):
            rgba = [c_uint() for _ in range(4)]
            if raw.FPDFPageObj_GetFillColor(obj, *(byref(value) for value in rgba)) and 0 < rgba[3].value <= 64:
                x0, bottom, x1, top = obj.get_bounds()
                if top - bottom >= 12:
                    boxes.append((x0, page.height - top, x1, page.height - bottom))
        return boxes


def extract_lines(page, faded=()):
    visible = page.filter(lambda obj: obj["object_type"] != "char" or (
        obj.get("upright", True) and not (obj["size"] >= 18 and (
            pale(obj) or any(inside(obj, box) for box in faded)
        ))
    ))
    words = visible.dedupe_chars(extra_attrs=("fontname", "size")).extract_words(extra_attrs=["fontname", "size"])
    tables = [Line(table_markdown(table.extract()), tuple(table.bbox), kind="table")
              for table in visible.find_tables()]
    tables = [table for table in tables if table.text]
    body_words = [word for word in words if not any(inside(word, table.box) for table in tables)]
    rows = text_rows(body_words)
    size = median([word["size"] for word in body_words]) if body_words else 10
    groups = [word_groups(row, max(15, size * 1.7)) for row in rows]
    inferred = list(aligned_tables(groups))
    lines = [word_line(group) for row in groups for group in row
             if not any(inside(group[0], table.box) for table in inferred)]
    return lines + tables + inferred


def column_gutter(lines, width):
    text = [line for line in lines if line.kind == "paragraph"]
    candidates = []
    for position in range(25, 76):
        x = width * position / 100
        left = [line for line in text if line.box[2] + 3 < x]
        right = [line for line in text if line.box[0] - 3 > x]
        if min(len(left), len(right)) < 2:
            continue
        if max(min(line.box[1] for line in side) for side in (left, right)) >= min(
            max(line.box[3] for line in side) for side in (left, right)
        ):
            continue
        crosses = len(text) - len(left) - len(right)
        if crosses <= max(2, len(text) // 5):
            gap = min(line.box[0] for line in right) - max(line.box[2] for line in left)
            candidates.append((crosses, -gap, abs(position - 50), x))
    return min(candidates)[-1] if candidates else None


def reading_order(lines, width):
    order = lambda line: (line.box[1], line.box[0])
    gutter = column_gutter(lines, width)
    if gutter is None:
        yield from ((0, line) for line in sorted(lines, key=order))
        return
    left, right = [], []
    for line in sorted(lines, key=order):
        if line.box[0] < gutter < line.box[2]:
            yield from ((1, item) for item in left)
            yield from ((2, item) for item in right)
            left, right = [], []
            yield 0, line
        elif line.box[2] <= gutter:
            left.append(line)
        else:
            right.append(line)
    yield from ((1, item) for item in left)
    yield from ((2, item) for item in right)


def noise_key(text):
    value = normalize(text, pdf=True).text.strip()
    return re.sub(r"\s+", "", re.sub(r"\d+", "#", value))


def repeated_noise(pages):
    occurrences = Counter()
    for _, _, height, lines in pages:
        occurrences.update({key for line in lines if line.kind == "paragraph"
                            and (line.box[1] < height * 0.09 or line.box[3] > height * 0.91 or line.size >= 24)
                            and (key := noise_key(line.text))})
    return {key for key, count in occurrences.items() if count >= max(2, (len(pages) + 1) // 2)}


def heading_level(line, body_size, levels):
    if line.level is not None:
        return line.level
    if line.bold and line.size >= body_size * 1.2 and len(line.text) <= 60:
        return min(range(len(levels)), key=lambda index: abs(levels[index] - line.size)) + 1
    return None


def page_blocks(lines, width, body_size, levels, *, layout_order=False):
    if layout_order:
        yield from lines
        return
    pending = []
    column = None

    def paragraph():
        return Line("\n".join(line.text.strip() for line in pending), bounds([line.box for line in pending]),
                    max(line.size for line in pending))

    for lane, line in reading_order(lines, width):
        level = heading_level(line, body_size, levels)
        atomic = line.kind != "paragraph" or level is not None
        if pending:
            previous = pending[-1]
            gap = line.box[1] - previous.box[3]
            if atomic or lane != column or gap > max(5, (previous.box[3] - previous.box[1]) * 0.85):
                yield paragraph()
                pending = []
        if level is not None:
            yield Line(line.text, line.box, line.size, line.bold, "heading", level)
        elif atomic:
            yield line
        else:
            pending.append(line)
            column = lane
    if pending:
        yield paragraph()


class HtmlTable(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in {"td", "th"} and self.rows:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.rows[-1].append("".join(self.cell))
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


class PageModels:
    def __init__(self):
        self.ocr = None
        self.layout = None

    def paddle(self, image, scale):
        from paddleocr import PaddleOCR
        if self.ocr is None:
            self.ocr = PaddleOCR(lang="ch", use_doc_orientation_classify=False,
                                 use_doc_unwarping=False, use_textline_orientation=False)
        lines = []
        for result in self.ocr.predict(str(image)):
            for text, box in zip(result["rec_texts"], result["rec_boxes"], strict=True):
                lines.append(Line(str(text), tuple(float(value) / scale for value in box)))
        return lines

    def structure(self, image, scale):
        try:
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise ValueError("PP-StructureV3 requires PaddleOCR layout dependencies") from exc
        if self.layout is None:
            self.layout = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False,
                                        use_formula_recognition=True)
        lines = []
        for result in self.layout.predict(str(image)):
            data = result.json
            for block in data.get("res", data)["parsing_res_list"]:
                label = block["block_label"]
                text = block["block_content"]
                box = tuple(float(value) / scale for value in block["block_bbox"])
                if label in {"header", "footer", "number", "image"}:
                    continue
                if label == "table":
                    table = HtmlTable()
                    table.feed(text)
                    lines.append(Line(table_markdown(table.rows) or text, box, kind="table"))
                elif label in {"doc_title", "paragraph_title"}:
                    lines.append(Line(text, box, kind="heading", level=1 if label == "doc_title" else 2))
                elif text.strip():
                    lines.append(Line(text, box))
        return lines

    def read(self, page, layout=False):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "page.png"
            rendered = page.to_image(resolution=300)
            rendered.original.save(image)
            scale = rendered.original.width / page.width
            if layout:
                return self.structure(image, scale)
            try:
                lines = self.paddle(image, scale)
                if any(line.text.strip() for line in lines):
                    return lines
            except (ImportError, RuntimeError, ValueError, OSError) as error:
                LOGGER.info("PaddleOCR unavailable for page %d: %s", page.page_number, type(error).__name__)
            return tesseract(image, scale)


def tesseract(image, scale):
    executable = shutil.which("tesseract")
    if executable is None:
        raise ValueError("OCR requires PaddleOCR or Tesseract with the chi_sim and eng language data")
    try:
        result = subprocess.run(
            [executable, str(image), "stdout", "-l", "chi_sim+eng", "--psm", "3", "tsv"],
            capture_output=True, check=True, encoding="utf-8", timeout=180,
        )
    except (subprocess.SubprocessError, UnicodeError) as exc:
        raise ValueError("Tesseract failed. Check the chi_sim and eng language data") from exc
    if re.search(r"failed loading language|error opening data file", result.stderr, re.I):
        raise ValueError("Tesseract could not load chi_sim or eng language data")
    groups = defaultdict(list)
    for word in csv.DictReader(StringIO(result.stdout), delimiter="\t"):
        if word.get("text", "").strip():
            x, y, w, h = (int(word[field]) / scale for field in ("left", "top", "width", "height"))
            key = tuple(word[field] for field in ("block_num", "par_num", "line_num"))
            groups[key].append({"text": word["text"], "x0": x, "top": y, "x1": x + w, "bottom": y + h})
    return [word_line(words) for words in groups.values()]


def parse_pdf(path):
    try:
        import pdfplumber
    except ImportError as exc:
        raise ValueError("PDF support requires pip install '.[documents]'") from exc
    layout = os.getenv("VGRAG_PDF_LAYOUT") == "pp-structure"
    models = PageModels()
    pages = []
    import pypdfium2

    with pdfplumber.open(path) as document, pypdfium2.PdfDocument(path) as rendered:
        for page in document.pages:
            try:
                lines = extract_lines(page, transparent_text(rendered, page)) if not layout else []
                if layout or sum(len(re.sub(r"\s", "", line.text)) for line in lines) < MIN_TEXT_CHARS:
                    if layout or page.chars or page.images or page.curves or page.rects:
                        ocr_lines = models.read(page, layout=layout)
                        if not ocr_lines and not lines:
                            raise ValueError("OCR returned no readable text")
                        lines = ocr_lines or lines
                pages.append((page.page_number, page.width, page.height, lines))
            except ValueError as exc:
                raise ValueError(f"PDF page {page.page_number}: {exc}") from exc
            finally:
                page.close()
    noise = repeated_noise(pages)
    sizes = [line.size for _, _, _, lines in pages for line in lines if line.size and line.kind == "paragraph"]
    body_size = median(sizes) if sizes else 10
    levels = []
    for size in sorted({size for size in sizes if size >= body_size * 1.2}, reverse=True):
        if not levels or levels[-1] - size >= 0.8:
            levels.append(size)
    builder = BlockBuilder("pdf")
    for number, width, height, lines in pages:
        body = [line for line in lines if not (
            noise_key(line.text) in noise and
            (line.box[1] < height * 0.09 or line.box[3] > height * 0.91 or line.size >= 24)
        ) and not (
            (line.box[1] < height * 0.09 or line.box[3] > height * 0.91)
            and re.fullmatch(r"(?:第\s*)?\d+(?:\s*页)?", line.text.strip())
        )]
        for index, block in enumerate(page_blocks(body, width, body_size, levels, layout_order=layout)):
            prior = builder.blocks[-1] if builder.blocks else None
            continues = bool(index == 0 and prior and prior.page_no == number - 1
                             and prior.block_type == block.kind == "paragraph"
                             and prior.text.rstrip()[-1:] not in SENTENCE_END)
            dehyphenated = bool(continues and re.search(r"[A-Za-z]-$", prior.text) and re.match(r"[a-z]", block.text))
            if dehyphenated:
                builder.blocks[-1] = replace(prior, content=prior.content.part(0, -1))
            builder.add(block.text, block.kind, f"第 {number} 页", level=block.level, page_no=number,
                        bbox=block.box, continues=continues, join_without_space=dehyphenated)
    return builder.blocks
