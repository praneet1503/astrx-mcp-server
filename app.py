from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import os

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"  # this will be in .gitignore in public repo

# cache
_cached_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cached_df
    try:
        # load if data exists; otherwise keep None
        # do NOT fail startup if data missing
        if DATA_DIR.exists():
            data_files = list((DATA_DIR).glob("*.csv")) + list((DATA_DIR).glob("*.json"))
            if data_files:
                # pick primary CSV/JSON (customize if you have naming)
                f = data_files[0]
                print(f"Loading data from {f}...")
                if f.suffix == ".csv":
                    _cached_df = pd.read_csv(f)
                else:
                    try:
                        _cached_df = pd.read_json(f)
                    except ValueError:
                        _cached_df = pd.read_json(f, lines=True)
                print(f"Loaded {_cached_df.shape[0]} records.")
    except Exception as e:
        _cached_df = None
        print("Warning: failed to load local data:", e)

    # Log public URL for HF Spaces
    space_host = os.getenv("SPACE_HOST")
    if space_host:
        print(f"Public API URL: https://{space_host}")
        print(f"Swagger UI: https://{space_host}/docs")
    else:
        print("Local URL: http://localhost:7860")

    yield
    _cached_df = None

app = FastAPI(title="Scraped Data API", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status":"UP"}

@app.get("/data")
def data_preview(limit: int = 100):
    if _cached_df is None:
        raise HTTPException(status_code=404, detail="No data available on the server")
    return _cached_df.head(limit).where(pd.notnull(_cached_df), None).to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
