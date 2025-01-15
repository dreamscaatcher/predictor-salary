# Stage 1: Build frontend
FROM node:20.11.0-bullseye-slim AS frontend-builder
WORKDIR /app

# Install dependencies
COPY frontend/package*.json frontend/
WORKDIR /app/frontend
RUN npm install --no-audit --no-fund

# Build frontend
COPY frontend .
RUN npm run build

# Stage 2: Build backend
FROM python:3.11.7-slim-bullseye AS backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and ML artifacts
COPY backend/app ./app
COPY backend/data ./data
COPY backend/model ./model

# Copy frontend static files
COPY --from=frontend-builder /app/frontend/out ./static

# Set environment variables
ENV PYTHONPATH=/app \
    PORT=8000 \
    PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
