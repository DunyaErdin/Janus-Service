FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system janus && adduser --system --ingroup janus janus

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --upgrade pip && pip install .

USER janus

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; port=os.getenv('EDGE_AI_SERVER_PORT', '8080'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)"

CMD ["janus-edge-ai"]
