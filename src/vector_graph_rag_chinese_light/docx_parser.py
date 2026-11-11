"""Read the DOCX body and footnotes, including OOXML structures outside paragraph runs."""

import re
from collections import defaultdict
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .documents import BlockBuilder, table_markdown

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
IGNORED = {W + "del", W + "moveFrom", W + "comment", W + "pPr", W + "rPr"}


def child(element, name):
    return element.find(W + name) if element is not None else None


def attr(element, name="val", default=None):
    return element.get(W + name, default) if element is not None else default


def read_xml(archive, name, *, required=False):
    try:
        return ET.fromstring(archive.read("word/" + name + ".xml"))
    except KeyError:
        if required:
            raise ValueError(f"DOCX has no {name}.xml") from None
        return ET.Element("missing")


def elements(container, tags):
    for element in container:
        if element.tag in IGNORED:
            continue
        if element.tag in tags:
            yield element
        else:
            yield from elements(element, tags)


def paragraph_parts(element):
    if element.tag in IGNORED:
        return
    if element.tag == W + "txbxContent":
        yield element
    elif element.tag == W + "t":
        yield element.text or ""
    elif element.tag in {W + "br", W + "cr"}:
        yield "\n"
    elif element.tag == W + "tab":
        yield "\t"
    elif element.tag == W + "footnoteReference":
        yield f"[^{attr(element, 'id')}]"
    elif element.tag.endswith(("}docPr", "}cNvPr", "}imagedata")):
        if alternative := (element.get("descr") or element.get("title")
                           or next((v for k, v in element.attrib.items() if k.endswith("}title")), "")):
            yield f"[图片：{alternative}]"
    else:
        for node in element:
            yield from paragraph_parts(node)


def table_rows(table):
    vertical = {}
    for row in elements(table, {W + "tr"}):
        before = int(attr(child(child(row, "trPr"), "gridBefore"), default=0))
        columns = [""] * before
        merged = {}
        for cell in elements(row, {W + "tc"}):
            properties = child(cell, "tcPr")
            width = int(attr(child(properties, "gridSpan"), default=1))
            value = "\n".join(
                "".join(paragraph_parts_in_cell(item)) if item.tag == W + "p"
                else table_markdown(list(table_rows(item)))
                for item in elements(cell, {W + "p", W + "tbl"})
            ).strip()
            merge = child(properties, "vMerge")
            for column in range(len(columns), len(columns) + width):
                if merge is None or attr(merge) == "restart":
                    resolved = value
                else:
                    resolved = "\n".join(dict.fromkeys(filter(None, (vertical.get(column), value))))
                columns.append(resolved)
                if merge is not None:
                    merged[column] = resolved
        vertical = merged
        yield columns


def paragraph_parts_in_cell(element):
    for part in paragraph_parts(element):
        if isinstance(part, str):
            yield part
        else:
            for paragraph in elements(part, {W + "p"}):
                yield from paragraph_parts_in_cell(paragraph)


class DocxStructure:
    def __init__(self, styles, numbering):
        self.styles = {attr(node, "styleId"): node for node in styles.findall(W + "style")}
        self.abstract = {attr(node, "abstractNumId"): node for node in numbering.findall(W + "abstractNum")}
        self.numbers = {attr(node, "numId"): node for node in numbering.findall(W + "num")}
        self.counters = defaultdict(dict)

    def paragraph_style(self, paragraph):
        properties = child(paragraph, "pPr")
        style = attr(child(properties, "pStyle"))
        numbering = child(properties, "numPr")
        outline = attr(child(properties, "outlineLvl"))
        visited = set()
        while style and style not in visited:
            visited.add(style)
            definition = self.styles.get(style)
            style_properties = child(definition, "pPr")
            if numbering is None:
                numbering = child(style_properties, "numPr")
            if outline is None:
                outline = attr(child(style_properties, "outlineLvl"))
            names = (style, attr(child(definition, "name"), default=""))
            level = next((match[1] for name in names
                          if (match := re.fullmatch(r"(?:heading|标题)\s*(\d+)", name, re.I))), None)
            if outline is None and level is not None:
                outline = str(int(level) - 1)
            style = attr(child(definition, "basedOn"))
        level = int(outline) + 1 if outline is not None and int(outline) < 9 else None
        return level, numbering

    def numbered(self, numbering, text):
        number_id = attr(child(numbering, "numId"))
        if not number_id or number_id == "0":
            return None
        level = int(attr(child(numbering, "ilvl"), default=0))
        instance = self.numbers.get(number_id)
        definition = self.abstract.get(attr(child(instance, "abstractNumId")))
        counters = self.counters[number_id]

        def number_properties(index):
            instance_level = next((node for node in instance.findall(W + "lvlOverride")
                                   if attr(node, "ilvl") == str(index)), None) if instance is not None else None
            specification = child(instance_level, "lvl")
            if specification is None and definition is not None:
                specification = next((node for node in definition.findall(W + "lvl")
                                      if attr(node, "ilvl") == str(index)), None)
            start = attr(child(instance_level, "startOverride"), default=attr(child(specification, "start"), default=1))
            return specification, int(start)

        specification, start = number_properties(level)
        counters[level] = counters.get(level, start - 1) + 1
        for index in list(counters):
            if index > level:
                del counters[index]

        def label(match):
            index = int(match[1]) - 1
            spec, initial = number_properties(index)
            value = counters.get(index, initial)
            style = attr(child(spec, "numFmt"))
            if style in {"lowerLetter", "upperLetter"} and value > 0:
                letters = ""
                while value:
                    value, letter = divmod(value - 1, 26)
                    letters = chr(ord("a") + letter) + letters
                return letters.upper() if style == "upperLetter" else letters
            if style == "decimalZero":
                return f"{value:02d}"
            return str(value)

        if attr(child(specification, "numFmt")) == "bullet":
            prefix = "-"
        else:
            prefix = re.sub(r"%([1-9])", label, attr(child(specification, "lvlText"), default=f"%{level + 1}."))
        return f"{'  ' * level}{prefix} {text}"


def parse_docx(path):
    with ZipFile(path) as archive:
        document = read_xml(archive, "document", required=True)
        footnotes = read_xml(archive, "footnotes")
        structure = DocxStructure(read_xml(archive, "styles"), read_xml(archive, "numbering"))
    builder = BlockBuilder("docx")
    paragraph_number = 0

    def emit(container, footnote=None):
        nonlocal paragraph_number
        for node in elements(container, {W + "p", W + "tbl"}):
            paragraph_number += 1
            location = f"脚注 {footnote}" if footnote is not None else f"第 {paragraph_number} 段"
            if node.tag == W + "tbl":
                builder.add(table_markdown(list(table_rows(node))), "table", location)
                continue
            parts = list(paragraph_parts(node))
            text = "".join(part for part in parts if isinstance(part, str)).strip()
            level, numbering = structure.paragraph_style(node)
            listed = structure.numbered(numbering, text) if text and level is None else None
            if footnote is not None and text:
                text = f"[^{footnote}] {text}"
            builder.add(listed or text, "heading" if level else "list" if listed else "paragraph",
                        location, level=level)
            for part in parts:
                if isinstance(part, ET.Element):
                    emit(part, footnote=footnote)

    body = child(document, "body")
    if body is None:
        raise ValueError("DOCX has no body")
    emit(body)
    builder.headings.clear()
    for footnote in footnotes.findall(W + "footnote"):
        if attr(footnote, "type") not in {"separator", "continuationSeparator", "continuationNotice"}:
            emit(footnote, footnote=attr(footnote, "id"))
    return builder.blocks
