# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthcheck if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch & requirements first for layer caching
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-cpu.txt

# Create a non-root user (Hugging Face Spaces runs with UID 1000)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application source code and models
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# Switch to non-root user
USER appuser

# Expose port 7860 (Hugging Face Spaces standard)
EXPOSE 7860

# Start FastAPI server via uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]
