from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
import os

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"  # this will be in .gitignore in public repo

app = FastAPI(title="Scraped Data API")

# cache
_cached_df = None

@app.on_event("startup")
def load_data():
    global _cached_df
    try:
        # load if data exists; otherwise keep None
        # do NOT fail startup if data missing
        if not DATA_DIR.exists():
             _cached_df = None
             return
             
        data_files = list((DATA_DIR).glob("*.csv")) + list((DATA_DIR).glob("*.json"))
        if not data_files:
            _cached_df = None
            return
        # pick primary CSV/JSON (customize if you have naming)
        f = data_files[0]
        if f.suffix == ".csv":
            _cached_df = pd.read_csv(f)
        else:
            _cached_df = pd.read_json(f, lines=True)
    except Exception as e:
        _cached_df = None
        print("Warning: failed to load local data:", e)

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
    return _cached_df.head(limit).to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
