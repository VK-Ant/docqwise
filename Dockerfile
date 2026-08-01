# ============================================
# DocQWise: Read. Extract. Retrieve.
# Multi-stage Docker build
# ============================================

# Stage 1: Base
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY docqwise/ docqwise/

# Stage 2: Core (minimal — PDF, DOCX, TXT, images)
FROM base AS core
RUN pip install --no-cache-dir .
CMD ["docqwise", "info"]

# Stage 3: ML (core + OCR, layout, embeddings)
FROM base AS ml
RUN pip install --no-cache-dir ".[ml]"
CMD ["docqwise", "info"]

# Stage 4: Full (all processing)
FROM base AS full
RUN pip install --no-cache-dir ".[full]"
CMD ["docqwise", "info"]

# Stage 5: Server (REST API + MCP)
FROM base AS server
RUN pip install --no-cache-dir ".[full,server]"
EXPOSE 8000 8080
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["docqwise", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 6: All (everything)
FROM base AS all
RUN pip install --no-cache-dir ".[all]"
EXPOSE 8000 8080
CMD ["docqwise", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Default: server stage
FROM server
