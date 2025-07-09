# syntax=docker/dockerfile:1.4

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        libglib2.0-0 \
        libsm6 \
        libsndfile1 \
        libxext6 \
        libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root user
RUN groupadd -r lexos && useradd --no-log-init -r -g lexos lexos

# --- Builder stage ---
FROM base AS builder

# Copy and install dependencies
COPY --link requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt

# Copy application code
COPY --link . .

# --- Final stage ---
FROM base AS final

WORKDIR /app

# Copy venv and application code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create runtime directories and set permissions
RUN mkdir -p /app/{lexos_memory,logs,static,temp,config,uploads,backups} && \
    chown -R lexos:lexos /app/{lexos_memory,logs,static,temp,config,uploads,backups}

USER lexos

# Expose ports
EXPOSE 8080 8081

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "main.py"]