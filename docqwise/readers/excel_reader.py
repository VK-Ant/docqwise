"""Excel reader (XLSX, XLS)."""
from __future__ import annotations
import hashlib, os
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class ExcelReader(BaseReader):
    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".xlsx", ".xls", ".xlsm", ".ods")

    def supported_formats(self) -> list[str]:
        return [".xlsx", ".xls", ".xlsm", ".ods"]

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        import openpyxl
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        sheets_filter = kwargs.get("sheets", None)
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        all_text = []
        all_rows = {}
        for sheet_name in wb.sheetnames:
            if sheets_filter and sheet_name not in sheets_filter:
                continue
            ws = wb[sheet_name]
            rows = []
            sheet_text = [f"Sheet: {sheet_name}"]
            headers = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                str_row = [str(c) if c is not None else "" for c in row]
                if i == 0:
                    headers = str_row
                rows.append(dict(zip(headers, str_row)) if headers else {f"col_{j}": v for j, v in enumerate(str_row)})
                sheet_text.append(" | ".join(str_row))
            all_text.extend(sheet_text)
            all_rows[sheet_name] = {"headers": headers, "rows": rows}
        wb.close()
        full_text = "\n".join(all_text)
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=DocumentType.EXCEL,
            metadata=DocumentMetadata(word_count=len(full_text.split()), file_size_bytes=file_size,
                                       file_hash=file_hash, has_selectable_text=True,
                                       custom={"sheets": all_rows}),
            text=full_text, pages=[PageContent(page_num=1, text=full_text)],
        )
