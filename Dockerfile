# Build backend
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
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
    fi && \
    echo "Model files created successfully at:" && \
    ls -l model/

# Expose port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting FastAPI application..."\n\
until $(curl --output /dev/null --silent --head --fail http://localhost:8000/health); do\n\
    echo "Waiting for FastAPI to start..."\n\
    sleep 1\n\
done\n\
echo "FastAPI is up and running"\n\
tail -f /dev/null' > /app/start.sh && \
    chmod +x /app/start.sh

# Start the backend with health check
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level debug & /app/start.sh
