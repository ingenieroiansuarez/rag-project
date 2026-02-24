# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app
# Install build tools just in case
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
# Copy project files before building
COPY . .

RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --target=/app/deps .[dev]

# Stage 2: Final image
FROM python:3.11-slim as runtime

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the installed dependencies from the builder
COPY --from=builder /app/deps /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/deps/bin/* /usr/local/bin/

# Copy application source code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Healthcheck configuration based on DoD
HEALTHCHECK --interval=8s --timeout=3s --retries=3 \
    CMD curl -sf http://localhost:8000/healthz || exit 1

# Entrypoint to start the FastAPI server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
