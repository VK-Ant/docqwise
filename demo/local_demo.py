"""
DocQWise: Process Your Local Documents
========================================

Point at any folder. DocQWise reads everything inside.
Search, extract, ask questions — every answer shows which document it came from.

Usage:
    python local_demo.py                        # uses demo/ folder
    python local_demo.py "C:/MyDocuments"       # your own folder

Requirements:
    pip install docqwise sentence-transformers
    ollama pull nemotron-mini
"""

import os
import sys
import tempfile
import shutil
from docqwise import Docqwise


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "demo"
    folder = os.path.abspath(folder)

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return

    print(f"DocQWise v0.3.0 — Local Document Intelligence")
    print(f"Folder: {folder}")
    print(f"=" * 50)
    print()

    db_path = os.path.join(tempfile.gettempdir(), "docqwise_local_demo")
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    dq = Docqwise(store_path=db_path)

    # Check LLM
    from docqwise.factory import LLMFactory
    backends = LLMFactory.available()
    if "ollama" in backends:
        print(f"LLM: Ollama connected")
    elif backends:
        print(f"LLM: {backends[0]}")
    else:
        print(f"LLM: none (install Ollama for best results)")
    print()

    # Step 1: Show files
    print("[1] FILES FOUND")
    EXTS = [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv",
            ".json", ".xml", ".yaml", ".yml", ".txt", ".md",
            ".html", ".pptx", ".eml", ".msg", ".jpg", ".jpeg",
            ".png", ".tiff", ".tif", ".parquet"]
    file_count = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                print(f"    {f}")
                file_count += 1
    print(f"    Total: {file_count} documents")
    print()

    # Step 2: Ingest
    print("[2] INGESTING")
    result = dq.ingest(folder, embed=False)
    print(f"    Processed: {result['processed']} files")
    print(f"    Time: {result['elapsed_s']}s")
    print()

    # Step 3: Extract from each document
    print("[3] EXTRACTION RESULTS")
    for root, dirs, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in [".pdf", ".docx", ".csv", ".json", ".xlsx"]:
                continue
            path = os.path.join(root, f)
            print(f"    --- {f} ---")

            labels = dq.classify(path)
            doc_type = labels[0]["label"] if labels else "unknown"
            print(f"    Type: {doc_type}")

            result = dq.extract_fields(path, model="nemotron-mini")
            if result.fields:
                for name, field in list(result.fields.items())[:8]:
                    val = str(field.value)[:45]
                    print(f"    {name:22s} = {val}")
            print()

    # Step 4: Interactive Q&A with source attribution
    print("[4] ASK YOUR DOCUMENTS")
    print("    Every answer shows which document it came from.")
    print("    Type a question. Type 'quit' to exit.")
    print()

    while True:
        try:
            query = input("    > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() == "quit":
            break

        # Get answer with source
        result = dq.ask_with_source(query)
        print(f"    Answer: {result.answer}")
        if result.source_name:
            print(f"    Source: {result.source_name}")
        if result.method:
            print(f"    Method: {result.method}")
        print()

    shutil.rmtree(db_path, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
