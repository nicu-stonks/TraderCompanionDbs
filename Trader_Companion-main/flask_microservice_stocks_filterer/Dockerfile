# Dockerfile for Flask microservice
FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY stocks_filtering_application/ ./stocks_filtering_application/
COPY api_endpoints.py .
COPY __init__.py .

EXPOSE 5000

# Set Python path to include the current directory
ENV PYTHONPATH=/app

CMD ["python", "api_endpoints.py"]