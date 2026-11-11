"""Token-budgeted splitting of the shared block representation."""

import logging
import re
from dataclasses import dataclass

from .documents import Block, LocatedText

LOGGER = logging.getLogger(__name__)
SPLITTER_VERSION = "structured-v1"
TERMINATORS = re.compile(r"[。！？；?!…]+|\.{2,}|\.")
CLOSERS = frozenset("”’」』）》】）\"')]}")
ABBREVIATION = re.compile(r"(?:\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|fig|no)|\b[a-z])\.$", re.I)
ATOMIC = {"code", "html", "quote", "list"}


@dataclass(frozen=True)
class Unit:
    content: LocatedText
    block: Block
    separator: str = ""

    @property
    def text(self):
        return self.content.text


@dataclass(frozen=True)
class Chunk:
    text: str
    location: str
    metadata: dict


def sentence_ranges(text):
    start = 0
    for match in TERMINATORS.finditer(text):
        end = match.end()
        if end <= start:
            continue
        if match[0] == ".":
            position = match.start()
            previous = text[position - 1:position] if position else ""
            following = text[end:end + 1]
            if ((previous.isascii() and previous.isalnum() and following.isascii() and following.isalnum())
                    or ABBREVIATION.search(text[max(0, end - 12):end])):
                continue
        while end < len(text) and text[end] in CLOSERS:
            end += 1
        if text[start:end].strip():
            yield start, end
        start = end
    if text[start:].strip():
        yield start, len(text)


def render(units):
    return "".join((unit.separator if index else "") + unit.text for index, unit in enumerate(units))


def make_chunk(units, count, *, text=None):
    first, last = units[0], units[-1]
    content = render(units) if text is None else text
    boxes = {}
    for unit in units:
        block = unit.block
        if block.bbox is not None:
            old = boxes.get(block.page_no, block.bbox)
            x0, y0, x1, y1 = block.bbox
            boxes[block.page_no] = (min(old[0], x0), min(old[1], y0), max(old[2], x1), max(old[3], y1))
    location = first.block.location
    if last.block.location != location:
        location += f"至{last.block.location}"
    metadata = {
        "heading_path": list(first.block.heading_path),
        "block_type": next((unit.block.block_type for unit in units if unit.block.block_type != "heading"), "heading"),
        "source_type": first.block.source_type,
        "page_no": first.block.page_no,
        "start_offset": first.content.offsets[0],
        "end_offset": last.content.offsets[-1] + 1,
        "token_count": count(content),
    }
    if last.block.page_no != first.block.page_no:
        metadata["end_page_no"] = last.block.page_no
    if boxes:
        if first.block.page_no in boxes:
            metadata["bbox"] = list(boxes[first.block.page_no])
        if len(boxes) > 1:
            metadata["page_bboxes"] = {str(page): list(box) for page, box in boxes.items()}
    return Chunk(content, location, metadata)


def characters(unit, count, maximum):
    LOGGER.warning("Character fallback at %s, offset %d (%d tokens)", unit.block.location,
                   unit.content.offsets[0], count(unit.text))
    text = unit.text
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if count(text[start:middle]) <= maximum:
                low = middle
            else:
                high = middle - 1
        part = unit.content.part(start, low).strip()
        if part.text:
            yield make_chunk([Unit(part, unit.block)], count)
        start = low


def table_chunks(block, count, maximum):
    unit = Unit(block.content, block)
    if count(unit.text) <= maximum:
        yield make_chunk([unit], count)
        return
    lines = list(re.finditer(r"[^\n]+", block.text))
    separator = next((i for i, line in enumerate(lines)
                      if i and re.fullmatch(r"[|:\- \t]+", line[0])), None)
    if separator is None or separator == len(lines) - 1:
        LOGGER.warning("Oversized atomic table at %s", block.location)
        yield make_chunk([unit], count)
        return
    header = block.text[:lines[separator].end()]
    pending = []
    for row in lines[separator + 1:]:
        unit = Unit(block.content.part(*row.span()), block, "\n")
        if pending and count(header + "\n" + render([*pending, unit])) > maximum:
            yield make_chunk(pending, count, text=header + "\n" + render(pending))
            pending = []
        pending.append(unit)
        if count(header + "\n" + unit.text) > maximum:
            LOGGER.warning("Oversized atomic table row at %s", block.location)
    if pending:
        yield make_chunk(pending, count, text=header + "\n" + render(pending))


def overlap_units(units, count, budget):
    suffix = []
    for unit in reversed(units):
        if unit.block.block_type == "heading" or count(render([unit, *suffix])) > budget:
            break
        suffix.insert(0, unit)
    return suffix


def split_prose(blocks, count, target, maximum, overlap):
    pending = []
    for block in blocks:
        sentences = []
        for start, end in sentence_ranges(block.text):
            content = block.content.part(start, end)
            if sentences:
                separator = content.text[:len(content.text) - len(content.text.lstrip())]
            elif block.continues:
                separator = " " if block.text[:1].isascii() and not block.join_without_space else ""
            else:
                separator = "\n\n"
            sentences.append(Unit(content.strip(), block, separator))
        if not sentences:
            continue
        # Try to keep the paragraph together before falling back to sentences.
        groups = [sentences] if count(render(sentences)) <= maximum else [[unit] for unit in sentences]
        for group in groups:
            if pending and count(render([*pending, *group])) > target:
                if count(render(pending)) >= 256 or count(render([*pending, *group])) > maximum:
                    yield make_chunk(pending, count)
                    pending = overlap_units(pending, count, round(target * overlap))
                    while pending and count(render([*pending, *group])) > maximum:
                        pending.pop(0)
            if count(render(group)) > maximum:
                yield from characters(group[0], count, maximum)
                pending = []
            else:
                pending.extend(group)
    if pending:
        yield make_chunk(pending, count)


def split_blocks(blocks, count, *, target=384, maximum=512, overlap=0.12):
    if not 0 < target <= maximum or not 0.10 <= overlap <= 0.15:
        raise ValueError("Invalid chunk token budget or sentence overlap")
    prose = []
    for block in blocks:
        boundary = prose and (
            block.heading_path != prose[-1].heading_path
            or (block.page_no != prose[-1].page_no and not block.continues)
        )
        if boundary or block.block_type in ATOMIC | {"table"}:
            yield from split_prose(prose, count, target, maximum, overlap)
            prose = []
        if block.block_type == "table":
            yield from table_chunks(block, count, maximum)
        elif block.block_type in ATOMIC:
            unit = Unit(block.content, block)
            if count(unit.text) > maximum:
                LOGGER.warning("Oversized atomic %s at %s", block.block_type, block.location)
            yield make_chunk([unit], count)
        else:
            prose.append(block)
    yield from split_prose(prose, count, target, maximum, overlap)
