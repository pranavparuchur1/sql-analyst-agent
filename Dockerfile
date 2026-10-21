FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Checkpoints live on a volume so conversations and paused approvals outlive the container.
    CHECKPOINT_DB=/state/checkpoints.db

WORKDIR /app
# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/__init__.py data/seed.py ./data/
COPY evals ./evals
# Seeding at build time bakes an identical, read-only sample DB into the image.
RUN python -m data.seed

RUN useradd --create-home agent && mkdir -p /state && chown agent /state
USER agent
VOLUME ["/state"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
