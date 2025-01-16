# Build backend
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
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

# Expose port
EXPOSE 8000

# Start the backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
