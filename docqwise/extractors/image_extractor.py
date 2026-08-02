"""Image/figure extraction from PDFs."""
from __future__ import annotations
from docqwise.core.document import DocqwiseDocument
from docqwise.core.element import ExtractedImage
from docqwise.core.chunk import BoundingBox

class ImageExtractor:
    def extract_images(self, document: DocqwiseDocument, **kwargs) -> list[ExtractedImage]:
        if document.doc_type.value != "pdf":
            return []
        try:
            import fitz
            doc = fitz.open(document.source_path)
            images = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                for img_idx, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        images.append(ExtractedImage(
                            image_id=f"p{page_num+1}_i{img_idx+1}",
                            page=page_num + 1,
                            image_bytes=base_image["image"],
                            resolution=(base_image.get("width", 0), base_image.get("height", 0)),
                            confidence=0.95,
                        ))
            doc.close()
            return images
        except Exception:
            return []
