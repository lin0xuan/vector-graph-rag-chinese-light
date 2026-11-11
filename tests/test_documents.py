"""Regression fixtures for structure, offsets, atomic blocks and PDF layout."""

import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from urllib.request import Request, urlopen

from vector_graph_rag_chinese_light.documents import parse_file, parse_markdown, parse_text
from vector_graph_rag_chinese_light.index import Index
from vector_graph_rag_chinese_light.splitter import sentence_ranges, split_blocks


class SplitterTests(unittest.TestCase):
    def test_gbk_and_original_offsets(self):
        original = "第一章 审批规则\r\n\r\n业务需经值守岗\r\n审批。"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.txt"
            path.write_bytes(original.encode("gbk"))
            blocks = parse_file(path)
        self.assertEqual(blocks[0].block_type, "heading")
        self.assertEqual(blocks[1].text, "业务需经值守岗审批。")
        chunk = list(split_blocks(blocks, len))[0]
        self.assertEqual(chunk.metadata["heading_path"], ["第一章 审批规则"])
        self.assertEqual(chunk.metadata["start_offset"], 0)
        self.assertEqual(chunk.metadata["end_offset"], len(original))
        self.assertEqual(blocks[1].content.offsets[0], original.index("业务"))

    def test_markdown_atomic_blocks_and_table_rows(self):
        code = "```python\nprint('原样保留')\n" + "x = 1\n" * 12 + "```"
        table = "| 项目 | 结果 |\n| --- | --- |\n" + "\n".join(f"| {n:02d} | 已完成 |" for n in range(10))
        html = "<div>\u3000原样</div>"
        quote = "> 引文第一行\n> 引文第二行"
        source = ("# 第一章\n\n## 1.2 节\n\n正文含 ![图注](image.png) 和 [审批规则](rule.md)。\n\n"
                  + code + "\n\n" + table + "\n\n" + quote + "\n\n" + html)
        blocks = parse_markdown(source)
        with self.assertLogs("vector_graph_rag_chinese_light.splitter", level="WARNING"):
            chunks = list(split_blocks(blocks, len, target=42, maximum=64))
        self.assertIn("图注", "".join(block.text for block in blocks))
        self.assertIn("审批规则", "".join(block.text for block in blocks))
        self.assertIn(code, [chunk.text for chunk in chunks])
        self.assertIn(quote, [chunk.text for chunk in chunks])
        self.assertIn(html, [chunk.text for chunk in chunks])
        tables = [chunk for chunk in chunks if chunk.metadata["block_type"] == "table"]
        self.assertGreater(len(tables), 1)
        self.assertTrue(all(chunk.text.startswith("| 项目 | 结果 |\n| --- | --- |\n") for chunk in tables))
        self.assertTrue(all(chunk.metadata["token_count"] <= 64 for chunk in tables))
        self.assertTrue(all(chunk.metadata["heading_path"] == ["第一章", "1.2 节"] for chunk in tables))

    def test_sentence_boundaries_and_whole_sentence_overlap(self):
        text = '使用 Node.js v1.2.3，值为 3.14（e.g. 该设置）。他说：“完成了！”再确认。'
        self.assertEqual([text[a:b] for a, b in sentence_ranges(text)],
                         ['使用 Node.js v1.2.3，值为 3.14（e.g. 该设置）。', '他说：“完成了！”', '再确认。'])
        source = "甲" * 24 + "。" + "短句。" + "乙" * 24 + "。"
        chunks = list(split_blocks(parse_text(source), len, target=35, maximum=42))
        self.assertEqual([chunk.text for chunk in chunks], [source[:28], source[25:]])
        self.assertEqual(chunks[1].metadata["start_offset"], 25)

    def test_long_unpunctuated_sentence_logs_character_fallback(self):
        source = "甲乙丙丁" * 100
        with self.assertLogs("vector_graph_rag_chinese_light.splitter", level="WARNING") as logs:
            chunks = list(split_blocks(parse_text(source), len, target=50, maximum=64))
        self.assertIn("Character fallback", logs.output[0])
        self.assertEqual("".join(chunk.text for chunk in chunks), source)
        self.assertTrue(all(chunk.metadata["token_count"] <= 64 for chunk in chunks))
        for chunk in chunks:
            self.assertEqual(chunk.text, source[chunk.metadata["start_offset"]:chunk.metadata["end_offset"]])

    def test_txt_list_item_keeps_wrapped_content(self):
        chunks = list(split_blocks(parse_text("（1）审批说明\n  补充资料。\n（2）登记。"), len))
        self.assertEqual([chunk.text for chunk in chunks], ["（1）审批说明\n  补充资料。", "（2）登记。"])
        self.assertTrue(all(chunk.metadata["block_type"] == "list" for chunk in chunks))

    def test_old_sqlite_index_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            with closing(sqlite3.connect(Path(directory) / "index.sqlite3")) as db:
                db.executescript("""
                    CREATE TABLE passages (id TEXT PRIMARY KEY, source TEXT, location TEXT, text TEXT);
                    INSERT INTO passages VALUES ('old', 'doc', '第 1 行', '原内容');
                """)
            self.assertEqual(Index(directory).passages()[0].as_dict(),
                             {"id": "old", "source": "doc", "location": "第 1 行", "text": "原内容"})


class StructuredDocumentTests(unittest.TestCase):
    def test_same_prose_boundaries_across_four_formats(self):
        try:
            from docx import Document
            from reportlab.pdfgen import canvas
            import pdfplumber  # noqa: F401
        except ImportError:
            self.skipTest("Install the documents and test extras for format fixtures")
        paragraphs = (
            "The operations team approves the request before implementation.",
            "The audit team keeps the final signed record for review.",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in ("txt", "md"):
                (root / f"policy.{suffix}").write_text("\n\n".join(paragraphs), encoding="utf-8")
            word = Document()
            for paragraph in paragraphs:
                word.add_paragraph(paragraph)
            word.save(root / "policy.docx")
            pdf = canvas.Canvas(str(root / "policy.pdf"))
            for index, paragraph in enumerate(paragraphs):
                pdf.drawString(45, 700 - 32 * index, paragraph)
            pdf.save()
            for suffix in ("txt", "md", "docx", "pdf"):
                with self.subTest(format=suffix):
                    blocks = parse_file(root / f"policy.{suffix}")
                    self.assertEqual([block.text for block in blocks], list(paragraphs))
                    self.assertEqual([chunk.text for chunk in split_blocks(blocks, len)], ["\n\n".join(paragraphs)])

    def test_docx_headings_numbering_tables_textboxes_and_notes(self):
        from zipfile import ZipFile

        w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.docx"
            with ZipFile(path, "w") as doc:
                doc.writestr("word/document.xml", f'''<w:document xmlns:w="{w}"><w:body>
                    <w:p><w:pPr><w:pStyle w:val="ChineseHeading"/></w:pPr><w:r><w:t>审批流程</w:t></w:r></w:p>
                    <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="7"/></w:numPr></w:pPr>
                        <w:r><w:t>名单维护</w:t></w:r></w:p>
                    <w:tbl><w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr>
                        <w:p><w:r><w:t>合并表头</w:t></w:r></w:p></w:tc></w:tr>
                        <w:tr><w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>值守岗</w:t></w:r></w:p></w:tc>
                        <w:tc><w:p><w:r><w:t>张三</w:t></w:r></w:p></w:tc></w:tr>
                        <w:tr><w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>
                        <w:tc><w:p><w:r><w:t>李四</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
                    <w:p><w:r><w:pict><w:txbxContent><w:p><w:r><w:t>文本框补充内容</w:t></w:r></w:p></w:txbxContent>
                        </w:pict></w:r><w:ins><w:r><w:t>接受修订</w:t></w:r></w:ins>
                        <w:del><w:r><w:delText>旧内容</w:delText></w:r></w:del>
                        <w:r><w:footnoteReference w:id="1"/></w:r></w:p>
                    </w:body></w:document>''')
                doc.writestr("word/styles.xml", f'''<w:styles xmlns:w="{w}">
                    <w:style w:styleId="ChineseHeading"><w:name w:val="标题 2"/></w:style></w:styles>''')
                doc.writestr("word/numbering.xml", f'''<w:numbering xmlns:w="{w}">
                    <w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="1"><w:start w:val="1"/>
                    <w:numFmt w:val="decimal"/><w:lvlText w:val="%1.%2"/></w:lvl></w:abstractNum>
                    <w:num w:numId="7"><w:abstractNumId w:val="0"/></w:num></w:numbering>''')
                doc.writestr("word/footnotes.xml", f'''<w:footnotes xmlns:w="{w}"><w:footnote w:id="1">
                    <w:p><w:r><w:t>脚注正文</w:t></w:r></w:p></w:footnote></w:footnotes>''')
                doc.writestr("word/comments.xml", f'<w:comments xmlns:w="{w}"><w:t>禁止混入的批注</w:t></w:comments>')
                doc.writestr("word/header1.xml", f'<w:hdr xmlns:w="{w}"><w:t>禁止混入的页眉</w:t></w:hdr>')
            blocks = parse_file(path)
        self.assertEqual(blocks[1].heading_path, ("审批流程",))
        self.assertEqual(blocks[1].text, "  1.1 名单维护")
        self.assertIn("| 合并表头 | 合并表头 |", blocks[2].text)
        self.assertIn("| 值守岗 | 李四 |", blocks[2].text)
        text = "\n".join(block.text for block in blocks)
        self.assertIn("文本框补充内容", text)
        self.assertIn("[^1] 脚注正文", text)
        self.assertIn("接受修订", text)
        self.assertNotIn("旧内容", text)
        self.assertNotIn("禁止混入", text)

    def test_two_column_pdf_and_running_headers(self):
        try:
            from reportlab.pdfgen import canvas
            import pdfplumber  # noqa: F401
        except ImportError:
            self.skipTest("Install the documents and test extras for PDF fixtures")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "columns.pdf"
            pdf = canvas.Canvas(str(path), pagesize=(600, 760))
            for page in (1, 2):
                pdf.drawString(42, 735, "RUNNING HEADER")
                pdf.drawString(270, 25, str(page))
                for row in range(3):
                    pdf.drawString(42, 660 - 14 * row, f"LEFT {page}-{row}: Evidence on the left side.")
                    pdf.drawString(340, 660 - 14 * row, f"RIGHT {page}-{row}: Evidence on the right.")
                pdf.saveState()
                pdf.setFont("Helvetica-Bold", 35)
                pdf.setFillAlpha(0.12)
                pdf.drawString(210, 400, "FADED MARK")
                pdf.restoreState()
                pdf.showPage()
            pdf.save()
            blocks = parse_file(path)
        self.assertEqual([block.page_no for block in blocks], [1, 1, 2, 2])
        self.assertEqual([block.text[:4] for block in blocks], ["LEFT", "RIGH", "LEFT", "RIGH"])
        self.assertTrue(all(block.bbox for block in blocks))
        self.assertNotIn("RUNNING HEADER", "\n".join(block.text for block in blocks))
        self.assertNotIn("FADED MARK", "\n".join(block.text for block in blocks))

    def test_real_chinese_ocr_on_mixed_pdf(self):
        if not shutil.which("tesseract"):
            self.skipTest("Tesseract is required for the real scanned-page fixture")
        languages = subprocess.check_output(["tesseract", "--list-langs"], text=True).splitlines()
        if not {"chi_sim", "eng"} <= set(languages):
            self.skipTest("Install the chi_sim and eng OCR language data")
        try:
            from PIL import Image, ImageDraw, ImageFont
            from reportlab.lib.utils import ImageReader
            from reportlab.pdfgen import canvas
            import pdfplumber  # noqa: F401
        except ImportError:
            self.skipTest("Install the documents and test extras for PDF fixtures")
        candidates = [os.getenv("VGRAG_TEST_CJK_FONT", ""),
                      "/System/Library/Fonts/STHeiti Medium.ttc",
                      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
        font = next((value for value in candidates if value and Path(value).is_file()), None)
        if font is None:
            self.skipTest("Set VGRAG_TEST_CJK_FONT to a Chinese font file")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.pdf"
            pdf = canvas.Canvas(str(path), pagesize=(600, 420))
            pdf.drawString(35, 360, "A sufficiently long text-layer paragraph that is extracted directly from this PDF.")
            pdf.showPage()
            image = Image.new("RGB", (1500, 1050), "white")
            draw = ImageDraw.Draw(image)
            typeface = ImageFont.truetype(font, 72)
            draw.text((80, 130), "审批流程由值守岗位负责。", font=typeface, fill="black")
            draw.text((80, 255), "请保留原始文件和审核记录。", font=typeface, fill="black")
            pdf.drawImage(ImageReader(image), 0, 0, width=600, height=420)
            pdf.save()
            blocks = parse_file(path)
            if chroma_url := os.getenv("VGRAG_TEST_CHROMA_URL"):
                from vector_graph_rag_chinese_light import ChineseLightRAG
                from vector_graph_rag_chinese_light.clients import COLLECTIONS

                rag = ChineseLightRAG(Path(directory) / "index", chroma_url=chroma_url, api_key="")
                try:
                    rag.upsert_file(path)
                    citations = [passage.as_dict() for passage in rag.index.passages()]
                    self.assertTrue(any(item["page_no"] == 2 and "审核记录" in item["text"] for item in citations))
                finally:
                    for kind in ("passages", "graph"):
                        name = f"vgrag-zh-{rag.index.namespace}-{kind}"
                        with urlopen(Request(f"{chroma_url}{COLLECTIONS}/{name}", method="DELETE"), timeout=15):
                            pass
        self.assertIn("text-layer paragraph", blocks[0].text)
        scanned = "".join(block.text for block in blocks if block.page_no == 2)
        self.assertIn("审批流程", scanned)
        self.assertIn("审核记录", scanned)
        self.assertNotIn("\ufffd", scanned)
        self.assertTrue(all(block.bbox for block in blocks))


if __name__ == "__main__":
    unittest.main()
