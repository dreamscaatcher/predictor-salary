# Stage 1: Build frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app

# Copy frontend files
COPY frontend/package*.json frontend/
WORKDIR /app/frontend
RUN npm install

COPY frontend .
RUN npm run build

# Stage 2: Build backend
FROM python:3.11-slim AS backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and ML artifacts
COPY backend/app ./app
COPY backend/data ./data
COPY backend/model ./model

# Copy frontend static files from builder
COPY --from=frontend-builder /app/frontend/out ./static

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
