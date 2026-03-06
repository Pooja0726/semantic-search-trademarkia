# ── Dockerfile ─────────────────────────────────────────────────────────────
# Bonus: Containerised FastAPI service
#
# Build:  docker build -t semantic-search .
# Run:    docker run -p 8000:8000 -v $(pwd)/data:/app/data semantic-search
#
# The /app/data volume mount allows the container to use pre-built data files
# (embeddings, ChromaDB, FCM results) from the host.  Without it the container
# would need to re-run setup.py on every start (~15 min).
#
# For a self-contained image that runs setup on first boot, remove the VOLUME
# line and add `RUN python setup.py` before the CMD.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies needed by umap-learn / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/     ./app/
COPY scripts/ ./scripts/
COPY setup.py .

# Create data directory (will be populated by volume mount or setup.py)
RUN mkdir -p ./data

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]