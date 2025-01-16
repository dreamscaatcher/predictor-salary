# Use Python slim image
FROM python:3.11.7-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and ML artifacts
COPY backend/app ./app/
COPY backend/data ./data/
COPY backend/model ./model/

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port (Railway will set the PORT environment variable)
EXPOSE ${PORT}

# Start FastAPI server (Railway will set the PORT)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 4
