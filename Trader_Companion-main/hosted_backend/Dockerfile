FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Create directory for database
RUN mkdir -p /app/data

# Expose the port Railway expects (defaults to 8080)
EXPOSE 8080

# Run the application using Railway's PORT env var when provided
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
