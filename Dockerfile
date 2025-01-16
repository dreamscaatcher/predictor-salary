# Build frontend
FROM node:18-alpine AS frontend-builder

# Set working directory
WORKDIR /app/frontend

# Copy package files first
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy the frontend source code
COPY frontend/ .

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
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data directory first
COPY backend/data ./data

# Copy model directory
COPY backend/model ./model

# Copy remaining backend code
COPY backend/ .

# Train the model and verify it exists
RUN python -c "from app.main import load_or_train_model; load_or_train_model()" && \
    if [ ! -f "model/lin_regress.sav" ] || [ ! -f "model/exp_encoder.sav" ] || [ ! -f "model/size_encoder.sav" ]; then \
        echo "Model files not created successfully" && exit 1; \
    fi

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/.next ./.next
COPY --from=frontend-builder /app/frontend/node_modules ./node_modules
COPY --from=frontend-builder /app/frontend/package.json ./package.json

# Expose ports
EXPOSE 3000 8000

# Create a script to start both services
RUN echo '#!/bin/bash\n\
npm start & \
uvicorn app.main:app --host 0.0.0.0 --port 8000\n\
wait' > start.sh && chmod +x start.sh

# Start both services
CMD ["./start.sh"]
