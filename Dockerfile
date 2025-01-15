# Build frontend
FROM node:18-alpine AS frontend-builder
# Set working directory
WORKDIR /app/frontend
# Copy package files first
COPY frontend/package.json frontend/package-lock.json ./
# Install dependencies with specific flags to avoid peer dependency issues
RUN npm install --legacy-peer-deps --force
# Copy the frontend source code
COPY frontend/ ./
# Set environment variables for production build
ENV NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NODE_ENV=production
# Build the frontend
RUN npm run build

# Build backend
FROM python:3.12-slim
# Set working directory
WORKDIR /app
# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Copy backend requirements
COPY backend/requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy backend code
COPY backend/ .
# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/out ./static
# Set environment variables
ENV PORT=8000
# Expose port
EXPOSE 8000
# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
