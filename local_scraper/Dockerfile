FROM python:3.10-slim

WORKDIR /app

COPY requirements-scraper.txt .
RUN pip install --no-cache-dir -r requirements-scraper.txt

# We don't need local browsers since we run on Modal
# RUN python -m playwright install --with-deps

COPY . .

# Entrypoint runs the scraper service (see docker-compose)
CMD ["python", "run_scraper_service.py"]
