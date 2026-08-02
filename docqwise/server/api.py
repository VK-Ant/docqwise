"""FastAPI REST server."""
from __future__ import annotations

def create_app(store_path: str = "./docqwise_db"):
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import JSONResponse
    import tempfile, os, shutil

    app = FastAPI(title="DocQWise API", description="Read. Extract. Retrieve.", version="0.2.0")

    from docqwise import Docqwise
    dq = Docqwise(store_path=store_path)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.2.0", "metrics": dq.metrics()}

    @app.post("/extract")
    async def extract(file: UploadFile = File(...), template: str = Form(None)):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        try:
            content = await file.read()
            tmp.write(content)
            tmp.close()
            from docqwise.readers.auto_reader import AutoReader
            reader = AutoReader()
            doc = reader.read(tmp.name)
            from docqwise.extractors.field_extractor import RegexFieldExtractor
            extractor = RegexFieldExtractor()
            result = extractor.extract_fields(doc, schema=None)
            return JSONResponse({"doc_id": doc.doc_id, "fields": result.to_dict(),
                                "confidence": result.confidence, "doc_type": doc.doc_type.value})
        finally:
            os.unlink(tmp.name)

    @app.post("/extract/tables")
    async def extract_tables(file: UploadFile = File(...)):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        try:
            content = await file.read()
            tmp.write(content)
            tmp.close()
            from docqwise.readers.auto_reader import AutoReader
            from docqwise.extractors.table_extractor import RuleBasedTableExtractor
            doc = AutoReader().read(tmp.name)
            tables = RuleBasedTableExtractor().extract_tables(doc)
            return JSONResponse({"tables": [t.to_json() for t in tables], "count": len(tables)})
        finally:
            os.unlink(tmp.name)

    @app.post("/classify")
    async def classify(file: UploadFile = File(...)):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        try:
            content = await file.read()
            tmp.write(content)
            tmp.close()
            from docqwise.readers.auto_reader import AutoReader
            from docqwise.classifier.zero_shot import ZeroShotClassifier
            doc = AutoReader().read(tmp.name)
            labels = ZeroShotClassifier().classify(doc.text)
            return JSONResponse({"classifications": labels[:5]})
        finally:
            os.unlink(tmp.name)

    @app.post("/pii/detect")
    async def detect_pii(file: UploadFile = File(...)):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        try:
            content = await file.read()
            tmp.write(content)
            tmp.close()
            from docqwise.readers.auto_reader import AutoReader
            from docqwise.security.pii_detector import PIIDetector
            doc = AutoReader().read(tmp.name)
            matches = PIIDetector().detect(doc.text)
            return JSONResponse({"pii": [{"type": m.pii_type, "value": m.value} for m in matches]})
        finally:
            os.unlink(tmp.name)

    return app

def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 4, store_path: str = "./docqwise_db"):
    import uvicorn
    app = create_app(store_path)
    uvicorn.run(app, host=host, port=port, workers=workers)
