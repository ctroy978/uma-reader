FROM python:3.10-slim

WORKDIR /app

# Install dependencies for SQLCipher
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlcipher-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory
COPY . .

# Ensure the data directory exists for SQLite
RUN mkdir -p /app/data

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]