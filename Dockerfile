# Stage 1: Build frontend
FROM node:20.11.0-bullseye-slim AS frontend-builder
WORKDIR /build

# Install dependencies
COPY frontend/package*.json frontend/
WORKDIR /build/frontend
RUN npm install

# Build frontend
COPY frontend ./ 
RUN npm run build

# Stage 2: Build backend
FROM python:3.11.7-slim-bullseye AS backend-builder
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

# Stage 3: Final image with Nginx
FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy frontend build
COPY --from=frontend-builder /build/frontend/out /usr/share/nginx/html

# Copy backend
COPY --from=backend-builder /app /app

# Install Python and dependencies in final image
RUN apk add --no-cache python3 py3-pip

# Install backend dependencies
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app \
    PORT=8000 \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 80 8000

# Start both nginx and backend server
CMD nginx && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
