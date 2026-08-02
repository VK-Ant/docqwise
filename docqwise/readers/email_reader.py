"""Email reader (EML, MSG)."""
from __future__ import annotations
import email, hashlib, os
from email import policy
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class EmailReader(BaseReader):
    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".eml", ".msg")

    def supported_formats(self) -> list[str]:
        return [".eml", ".msg"]

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read()
            file_hash = hashlib.sha256(raw).hexdigest()[:16]
        msg = email.message_from_bytes(raw, policy=policy.default)
        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        to = msg.get("to", "")
        date = msg.get("date", "")
        body = msg.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body else ""
        header_text = f"From: {sender}\nTo: {to}\nDate: {date}\nSubject: {subject}\n\n"
        full_text = header_text + text
        attachments = []
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                attachments.append(part.get_filename() or "unnamed")
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=DocumentType.EMAIL,
            metadata=DocumentMetadata(title=subject, author=sender, word_count=len(full_text.split()),
                                       file_size_bytes=file_size, file_hash=file_hash,
                                       has_selectable_text=True,
                                       custom={"attachments": attachments, "to": to, "date": date}),
            text=full_text, pages=[PageContent(page_num=1, text=full_text)],
        )
