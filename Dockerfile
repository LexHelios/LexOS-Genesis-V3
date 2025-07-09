# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# --- Builder stage ---
FROM base AS builder

# Install system dependencies required for pip packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
        && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first for better caching
COPY --link requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --upgrade pip && \
    --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the rest of the application code (excluding .git, .env, etc.)
COPY --link . .

# --- Final stage ---
FROM base AS final

# Create a non-root user
RUN addgroup --system lexos && adduser --system --ingroup lexos lexos

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
# Copy application code from builder
COPY --from=builder /app /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set permissions
RUN chown -R lexos:lexos /app

USER lexos

# Expose the main service port (default 8080, can be overridden by env)
EXPOSE 8080

# Default command (can be overridden in docker-compose)
CMD ["python", "main.py"]
