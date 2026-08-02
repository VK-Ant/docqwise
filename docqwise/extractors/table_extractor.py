"""Table extraction from PDF documents."""
from __future__ import annotations
import re
from docqwise.extractors.base import BaseTableExtractor
from docqwise.core.document import DocqwiseDocument
from docqwise.core.element import Table, TableCell
from docqwise.core.chunk import BoundingBox

class RuleBasedTableExtractor(BaseTableExtractor):
    """Extract tables from PDFs using PyMuPDF built-in table detection."""

    def extract_tables(self, document: DocqwiseDocument, **kwargs) -> list[Table]:
        if document.doc_type.value != "pdf":
            return self._extract_from_text(document)
        try:
            import fitz
            doc = fitz.open(document.source_path)
            tables = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_tables = page.find_tables()
                for t_idx, tab in enumerate(page_tables):
                    cells = []
                    data = tab.extract()
                    rows = len(data)
                    cols = len(data[0]) if data else 0
                    for r, row in enumerate(data):
                        for c, val in enumerate(row):
                            cells.append(TableCell(
                                text=str(val) if val else "",
                                row=r, col=c,
                                is_header=(r == 0),
                            ))
                    table_id = f"p{page_num+1}_t{t_idx+1}"
                    tables.append(Table(
                        table_id=table_id, cells=cells,
                        rows=rows, cols=cols,
                        page=page_num + 1, confidence=0.90,
                    ))
            doc.close()
            return tables
        except Exception:
            return self._extract_from_text(document)

    def _extract_from_text(self, document: DocqwiseDocument) -> list[Table]:
        """Fallback: detect pipe-delimited or tab-delimited tables in text."""
        tables = []
        lines = document.text.split("\n")
        table_lines = []
        for line in lines:
            if "|" in line and line.count("|") >= 2:
                table_lines.append(line)
            elif table_lines:
                if len(table_lines) >= 2:
                    tables.append(self._parse_pipe_table(table_lines, len(tables) + 1))
                table_lines = []
        if len(table_lines) >= 2:
            tables.append(self._parse_pipe_table(table_lines, len(tables) + 1))
        return tables

    def _parse_pipe_table(self, lines: list[str], idx: int) -> Table:
        cells = []
        parsed_rows = []
        for line in lines:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if all(set(p) <= set("-:") for p in parts):
                continue
            parsed_rows.append(parts)
        cols = max(len(r) for r in parsed_rows) if parsed_rows else 0
        for r, row in enumerate(parsed_rows):
            for c, val in enumerate(row):
                cells.append(TableCell(text=val, row=r, col=c, is_header=(r == 0)))
        return Table(table_id=f"text_t{idx}", cells=cells,
                     rows=len(parsed_rows), cols=cols, confidence=0.75)
