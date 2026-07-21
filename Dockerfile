FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, models, and static files
COPY src/ /app/src/
COPY models/ /app/models/
COPY static/ /app/static/

# Expose port
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
