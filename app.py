import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import importlib
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Install Playwright browsers
try:
    logger.info("Installing Playwright browsers...")
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    logger.info("Playwright browsers installed successfully.")
except Exception as e:
    logger.error(f"Failed to install playwright browsers: {e}")

# Initialize FastAPI
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE_JSON = DATA_DIR / "animals.json"
DATA_FILE_CSV = DATA_DIR / "animals.csv"

# Global Cache
CACHED_DATA = None

def load_dataset():
    """
    Loads the dataset from JSON or CSV.
    Caches the result in memory.
    """
    global CACHED_DATA
    if CACHED_DATA is not None:
        return CACHED_DATA

    if DATA_FILE_JSON.exists():
        try:
            logger.info(f"Loading data from {DATA_FILE_JSON}")
            # Read JSON using pandas
            df = pd.read_json(DATA_FILE_JSON)
            CACHED_DATA = df.to_dict(orient="records")
            return CACHED_DATA
        except ValueError:
            # Fallback for simple list of dicts if pandas fails on format
            import json
            with open(DATA_FILE_JSON, 'r') as f:
                CACHED_DATA = json.load(f)
            return CACHED_DATA
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")

    if DATA_FILE_CSV.exists():
        try:
            logger.info(f"Loading data from {DATA_FILE_CSV}")
            df = pd.read_csv(DATA_FILE_CSV)
            CACHED_DATA = df.to_dict(orient="records")
            return CACHED_DATA
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")

    return {"error": "Dataset not found"}

@app.get("/")
def root():
    return "API is running"

@app.get("/health")
def health():
    return {"status": "UP"}

@app.get("/data")
def get_data():
    data = load_dataset()
    if isinstance(data, dict) and "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data

@app.get("/scrape")
def scrape_data():
    global CACHED_DATA
    try:
        logger.info("Starting scraper...")
        
        # Ensure current directory is in sys.path to import scraper
        if str(BASE_DIR) not in sys.path:
            sys.path.append(str(BASE_DIR))

        # Import scraper
        import scraper
        
        # Run the main function
        if hasattr(scraper, "main"):
             scraper.main()
        else:
            raise ImportError("No main() function found in scraper.py")

        # Clear cache to reload new data on next request
        CACHED_DATA = None
        
        return {"status": "success", "message": "Scraper executed successfully"}

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

