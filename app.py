from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import os
import gradio as gr
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_FILE = DATA_DIR / "animals.json"

# --- Configuration ---
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-haiku-20240307"

# --- Global State ---
_cached_df = None
ANIMALS_DATA = []

# --- Helper Functions ---
def load_data_list():
    """Loads the animals dataset from the JSON file as a list of dicts."""
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found.")
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} records from {DATA_FILE}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def search_animals(query):
    """
    Simple keyword search to find relevant animals.
    Returns a list of animal dictionaries.
    """
    query_words = query.lower().split()
    matches = []
    
    for animal in ANIMALS_DATA:
        # Search in name and description
        text = (animal.get("name", "") + " " + str(animal.get("description", ""))).lower()
        
        # Score based on how many query words appear
        score = sum(1 for word in query_words if word in text)
        
        if score > 0:
            matches.append((score, animal))
            
    # Sort by score (descending) and take top 20
    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:20]]

def query_claude(user_input):
    """
    Sends the user input and the dataset context to Claude.
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        return "Error: CLAUDE_API_KEY not found in environment variables."
    
    if not ANIMALS_DATA:
        return "Error: No animal data loaded. Please check the server logs."

    # Retrieval Step: Find relevant animals to reduce token count
    relevant_animals = search_animals(user_input)
    
    if not relevant_animals:
        context_str = "No specific animals found matching the query in the dataset."
    else:
        context_str = json.dumps(relevant_animals, ensure_ascii=False)

    print(f"Sending {len(relevant_animals)} records to Claude...")

    system_prompt = (
        "You are an expert zoologist assistant. "
        "You have access to a subset of the animal dataset relevant to the user's query below. "
        "Your task is to answer the user's question based strictly on this provided data. "
        "If the answer is not in the data, say so. "
        "Cite the 'name' of the animal when providing facts.\n\n"
        f"RELEVANT DATA:\n{context_str}"
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "No text returned.")
            else:
                return "Empty response from Claude."
        else:
            return f"API Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"Request Failed: {str(e)}"

# --- Gradio UI ---
def create_gradio_app():
    with gr.Blocks(title="Astrx MCP Server") as demo:
        gr.Markdown("# 🚀 Astrx MCP Server")
        gr.Markdown(
            "Ask questions about the animal dataset! "
            "This tool uses **Claude 3** to reason over the `animals.json` file."
        )
        
        with gr.Row():
            user_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., Which animals live in the desert? or Tell me about the Golden Retriever."
            )
            
        submit_btn = gr.Button("Ask Claude", variant="primary")
        output_box = gr.Textbox(label="Claude's Answer", lines=10)
        
        submit_btn.click(
            fn=query_claude,
            inputs=[user_input],
            outputs=output_box
        )
        
        gr.Markdown("---")
        gr.Markdown(f"**Dataset Status:** Loaded {len(ANIMALS_DATA)} records.")
    return demo

# --- FastAPI Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cached_df, ANIMALS_DATA
    try:
        # Load data for API (Pandas)
        if DATA_DIR.exists():
            data_files = list((DATA_DIR).glob("*.csv")) + list((DATA_DIR).glob("*.json"))
            if data_files:
                f = data_files[0]
                print(f"Loading data from {f}...")
                if f.suffix == ".csv":
                    _cached_df = pd.read_csv(f)
                else:
                    try:
                        _cached_df = pd.read_json(f)
                    except ValueError:
                        _cached_df = pd.read_json(f, lines=True)
                print(f"Loaded {_cached_df.shape[0]} records for API.")
        
        # Load data for Gradio (List of Dicts)
        ANIMALS_DATA = load_data_list()
        
    except Exception as e:
        _cached_df = None
        ANIMALS_DATA = []
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
    ANIMALS_DATA = []

# --- Main App ---
app = FastAPI(title="Astrx MCP Server", lifespan=lifespan)

# Mount Gradio App
# We mount it at the root "/" so it's the main interface
# The API docs will still be at /docs
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")

@app.get("/health")
def health():
    return {"status":"UP"}

@app.get("/api/data")
def data_preview(limit: int = 100):
    if _cached_df is None:
        raise HTTPException(status_code=404, detail="No data available on the server")
    return _cached_df.head(limit).where(pd.notnull(_cached_df), None).to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
