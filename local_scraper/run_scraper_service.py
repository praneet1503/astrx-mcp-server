import time
from datetime import datetime
from scraper import run_scrape_once
import logging
import os

logging.basicConfig(level=logging.INFO)

INTERVAL_SECONDS = int(os.environ.get('SCRAPER_INTERVAL', 3600))

def main():
    logging.info("Scraper service started, interval=%s", INTERVAL_SECONDS)
    while True:
        try:
            logging.info("Scrape started at %s", datetime.utcnow().isoformat())
            run_scrape_once()
            logging.info("Scrape finished at %s", datetime.utcnow().isoformat())
        except Exception as e:
            logging.exception("Scraper failed: %s", e)
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
