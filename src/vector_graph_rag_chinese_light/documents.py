"""Parse source documents into a shared sequence of source-located blocks."""

import codecs
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chardet import detect
from markdown_it import MarkdownIt

BlockType = Literal["heading", "paragraph", "code", "table", "list", "quote", "html"]
SourceType = Literal["txt", "md", "docx", "pdf"]
Box = tuple[float, float, float, float]
HAN = r"[\u3400-\u9fff\U00020000-\U000323af]"
SENTENCE_END = "。！？；…….!?;”’）】》"
TRANSPORT = re.compile(r"\r\n?|[\u3000\u200b\ufeff]")
BLANK_LINES = re.compile(r"\n(?:[^\S\n]*\n)+")
NUMBERED = re.compile(
    r"^(?:(第[一二三四五六七八九十百零〇\d]+[章节篇部])\s*|"
    r"([一二三四五六七八九十百零〇]+)[、.]\s*|"
    r"(\d+(?:\.\d+)+)(?:[.、\s]+|(?=" + HAN + r")|$))"
)
LIST_ITEM = re.compile(r"^(?:[（(][\d一二三四五六七八九十]+[）)]|\d+[.、](?!\d)|[-+*]\s)")


@dataclass(frozen=True)
class LocatedText:
    text: str
    offsets: tuple[int, ...]

    def part(self, start=0, end=None):
        return LocatedText(self.text[start:end], self.offsets[start:end])

    def strip(self):
        return self.part(len(self.text) - len(self.text.lstrip()), len(self.text.rstrip()))


@dataclass(frozen=True)
class Block:
    content: LocatedText
    heading_path: tuple[str, ...]
    block_type: BlockType
    order: int
    source_type: SourceType
    location: str
    page_no: int | None = None
    bbox: Box | None = None
    continues: bool = False
    join_without_space: bool = False

    @property
    def text(self):
        return self.content.text


def decode_text(data):
    encodings = ("utf-8-sig",) if data.startswith(codecs.BOM_UTF8) else ("utf-8", "utf-8-sig")
    encodings += ("gb18030", "gbk", "big5")
    for encoding in encodings:
        try:
            text = data.decode(encoding)
        except UnicodeError:
            continue
        if valid_decoding(text):
            return text
    guess = detect(data)
    if guess["encoding"] and guess["confidence"] >= 0.8:
        try:
            text = data.decode(guess["encoding"])
            if valid_decoding(text):
                return text
        except (LookupError, UnicodeError):
            pass
    raise ValueError("Could not decode document as UTF-8, GB18030, GBK, Big5 or a detected encoding")


def valid_decoding(text):
    return not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]", text)


def rewrite(content, pattern, replacement):
    parts, offsets = [], []
    cursor = 0
    for match in pattern.finditer(content.text):
        start, end = match.span()
        parts.extend((content.text[cursor:start], replacement(match)))
        offsets.extend(content.offsets[cursor:start])
        offsets.extend([content.offsets[start]] * len(parts[-1]))
        cursor = end
    parts.append(content.text[cursor:])
    offsets.extend(content.offsets[cursor:])
    return LocatedText("".join(parts), tuple(offsets))


def normalize(raw, start=0, *, pdf=False):
    content = LocatedText(raw, tuple(range(start, start + len(raw))))
    values = {"\r": "\n", "\r\n": "\n", "\u3000": " ", "\u200b": "", "\ufeff": ""}
    content = rewrite(content, TRANSPORT, lambda match: values[match[0]])
    if pdf:
        content = rewrite(content, re.compile(rf"(?<={HAN})[ \t]+(?={HAN})"), lambda _: "")
    return content


def repair_prose(content, *, hyphenated=False):
    if hyphenated:
        content = rewrite(content, re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])"), lambda _: "")
    content = rewrite(content, BLANK_LINES, lambda _: "\n\n")
    breaks = re.compile(rf"(?<=[^\s{SENTENCE_END}])[ \t]*\n[ \t]*(?={HAN})")
    return rewrite(content, breaks, lambda _: "").strip()


class BlockBuilder:
    def __init__(self, source_type):
        self.source_type = source_type
        self.blocks = []
        self.headings = {}
        self.offset = 0

    def add(self, content, kind, location, *, level=None, page_no=None, bbox=None, continues=False,
            join_without_space=False):
        if isinstance(content, str):
            start = self.offset
            self.offset += len(content) + 1
            content = normalize(content, start, pdf=self.source_type == "pdf")
        if kind == "paragraph":
            content = repair_prose(content, hyphenated=self.source_type == "pdf")
        elif kind not in {"code", "html", "list"}:
            content = content.strip()
        if not content.text.strip():
            return
        if level is not None:
            self.headings = {n: title for n, title in self.headings.items() if n < level}
            self.headings[level] = content.text.strip()
        self.blocks.append(Block(content, tuple(self.headings.values()), kind, len(self.blocks),
                                 self.source_type, location, page_no, bbox, continues, join_without_space))


def parse_text(raw):
    content = normalize(raw)
    builder = BlockBuilder("txt")
    lines = list(re.finditer(r"[^\n]*\n|[^\n]+$", content.text))
    last_content_line = max((n for n, line in enumerate(lines, 1) if line[0].strip()), default=0)
    paragraph_start = None
    paragraph_line = None
    paragraph_kind = "paragraph"
    for number, line in enumerate(lines, 1):
        value = line[0].rstrip("\n")
        stripped = value.strip()
        label = f"第 {number} 行"
        numbered = NUMBERED.match(stripped)
        following = lines[number][0].strip() if number < len(lines) else ""
        standalone = paragraph_start is None and (
            number == 1 or not lines[number - 2][0].strip()
        )
        title_boundary = (not following and number < last_content_line) or len(following) >= 25
        short_heading = (standalone and title_boundary and 0 < len(stripped) <= 24
                         and not re.search(r"[，、：,:“‘]", stripped)
                         and stripped[-1:] not in SENTENCE_END)
        kind, level = None, None
        if numbered:
            kind = "heading"
            level = 1 if numbered[1] else 2 if numbered[2] else numbered[3].count(".") + 2
        elif LIST_ITEM.match(stripped) or (
            value.startswith(("  ", "\t")) and stripped and
            (paragraph_start is None or paragraph_kind not in {"list", "quote"})
        ):
            kind = "quote" if stripped.startswith(("“", ">")) else "list"
        elif short_heading:
            kind, level = "heading", 1
        if (not stripped or kind) and paragraph_start is not None:
            fragment = content.part(paragraph_start, line.start())
            builder.add(fragment.part(0, len(fragment.text.rstrip())), paragraph_kind, f"第 {paragraph_line} 行")
            paragraph_start = None
        if kind == "heading":
            builder.add(content.part(line.start(), line.start() + len(value)), kind, label, level=level)
        elif stripped and paragraph_start is None:
            paragraph_start, paragraph_line = line.start(), number
            paragraph_kind = kind or "paragraph"
    if paragraph_start is not None:
        fragment = content.part(paragraph_start)
        builder.add(fragment.part(0, len(fragment.text.rstrip())), paragraph_kind, f"第 {paragraph_line} 行")
    return builder.blocks


def parse_markdown(raw):
    content = normalize(raw)
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = parser.parse(content.text)
    starts = [0, *(match.end() for match in re.finditer("\n", content.text)), len(content.text)]
    builder = BlockBuilder("md")
    types = {"paragraph_open": "paragraph", "fence": "code", "code_block": "code",
             "table_open": "table", "html_block": "html", "blockquote_open": "quote",
             "list_item_open": "list", "heading_open": "heading", "hr": "paragraph"}
    covered = 0
    for token in tokens:
        if token.type not in types or not token.map or token.map[0] < covered:
            continue
        first, last = token.map
        kind = types[token.type]
        fragment = content.part(starts[first], starts[last])
        if kind == "heading":
            if token.markup.startswith("#"):
                first_line = fragment.text.split("\n", 1)[0]
                title = re.sub(r"^ {0,3}#{1,6}(?:[ \t]+|$)|[ \t]+#+[ \t]*$", "", first_line).strip()
                begin = first_line.find(title)
                fragment = fragment.part(begin, begin + len(title))
            else:
                fragment = fragment.part(0, starts[last - 1] - starts[first]).strip()
        elif kind in {"code", "html"}:
            if fragment.offsets:
                original = fragment.offsets[0]
                fragment = LocatedText(raw[original:fragment.offsets[-1] + 1],
                                       tuple(range(original, fragment.offsets[-1] + 1)))
                fragment = rewrite(fragment, re.compile(r"\r\n?"), lambda _: "\n")
            if kind == "code":
                fragment = fragment.part(0, len(fragment.text.rstrip("\n")))
        else:
            fragment = fragment.part(0, len(fragment.text.rstrip("\n")))
        builder.add(fragment, kind, f"第 {first + 1} 行",
                    level=int(token.tag[1:]) if kind == "heading" else None)
        covered = last
    return builder.blocks


def table_markdown(rows):
    if not rows:
        return ""
    width = max(map(len, rows), default=0)
    if not width:
        return ""
    result = []
    for index, row in enumerate(rows):
        cells = [str(cell or "").strip().replace("\\", "\\\\").replace("|", "\\|")
                 .replace("\n", "<br>") for cell in row]
        result.append("| " + " | ".join(cells + [""] * (width - len(cells))) + " |")
        if index == 0:
            result.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(result)


def parse_file(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        parser = parse_text if suffix == ".txt" else parse_markdown
        return parser(decode_text(path.read_bytes()))
    if suffix == ".docx":
        from .docx_parser import parse_docx
        return parse_docx(path)
    if suffix == ".pdf":
        from .pdf_parser import parse_pdf
        return parse_pdf(path)
    raise ValueError(f"Unsupported document format: {suffix}")
