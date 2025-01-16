# Stage 1: Build frontend
FROM node:20.11.0-bullseye-slim AS frontend-builder

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY frontend/ ./

# Build the application
RUN npm run build

# Stage 2: Final image
FROM python:3.11.7-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy nginx configuration
RUN echo 'server {\n\
    listen $PORT;\n\
    root /app/static;\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
    location /api/ {\n\
        proxy_pass http://localhost:8000/;\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
    }\n\
    location = /health {\n\
        proxy_pass http://localhost:8000/health;\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
    }\n\
}' > /etc/nginx/conf.d/default.conf

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and ML artifacts
COPY backend/app ./app/
COPY backend/data ./data/
COPY backend/model ./model/

# Copy frontend build
COPY --from=frontend-builder /app/out /app/static

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Create start script
RUN echo '#!/bin/bash\n\
sed -i "s/\$PORT/$PORT/g" /etc/nginx/conf.d/default.conf\n\
nginx\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4' > /app/start.sh && \
chmod +x /app/start.sh

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE $PORT

# Start both nginx and backend server
CMD ["/app/start.sh"]
