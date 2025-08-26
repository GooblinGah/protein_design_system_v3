# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (basic build; add others as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy & install pinned deps
COPY requirements-pinned.txt ./
RUN pip install --no-cache-dir -r requirements-pinned.txt

# (Optional) TMalign or SignalP can be added via custom layers if licensed/available

# Copy project
COPY . .

# Default command shows help
CMD ["bash", "-lc", "echo 'Container ready. Try: python scripts/smoke_data.py && python train.py --epochs 1 --amp 0 --use_wandb 0'"]
