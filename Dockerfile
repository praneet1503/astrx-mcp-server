FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for numpy/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Expose the port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["python", "app.py"]
