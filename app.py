from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import os
import gradio as gr
import json
from dotenv import load_dotenv

# Import logic
from logic import run_model, set_animals_data, search_animals, initialize_retriever, save_keys

# Load environment variables
load_dotenv()

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_FILE = DATA_DIR / "animals.json"

# --- Global State ---
_cached_df = None

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

# --- Gradio UI ---
def create_gradio_app():
    with gr.Blocks(title="Astrx MCP Server") as demo:
        gr.Markdown("# 🚀 Astrx MCP Server")
        
        # --- Sponsor API Keys Panel ---
        with gr.Accordion("🔑 Sponsor API Keys (Session Only)", open=True):
            gr.Markdown("Enter your API keys below. They are stored in memory for this session only and are **not** saved to disk.")
            with gr.Row():
                samba_key = gr.Textbox(label="SambaNova Cloud API Key (Optional)", type="password", placeholder="Enter key...")
                claude_key = gr.Textbox(label="Anthropic Claude API Key (Necessary)", type="password", placeholder="Enter key...")
            with gr.Row():
                modal_key = gr.Textbox(label="Modal Token (ID:Secret) (Optional)", type="password", placeholder="e.g., ak-123:as-456")
                blaxel_key = gr.Textbox(label="Blaxel API Key (Optional)", type="password", placeholder="Enter key...")
            with gr.Row():
                gemini_key = gr.Textbox(label="Google Gemini API Key (Optional)", type="password", placeholder="Enter key...")
            
            save_keys_btn = gr.Button("Save Keys", variant="secondary")
            key_status = gr.Markdown("")

            save_keys_btn.click(
                fn=save_keys,
                inputs=[samba_key, claude_key, modal_key, blaxel_key, gemini_key],
                outputs=[key_status]
            )

        gr.Markdown(
            "Ask questions about the animal dataset! "
            "Choose a model provider below to reason over the `animals.json` file."
        )
        
        # --- Model Selection ---
        model_provider = gr.Dropdown(
            label="Choose Model Provider",
            choices=[
                "SambaNova (Optional) – Llama 3.1 8B",
                "SambaNova (Optional) – Llama 3.1 70B",
                "SambaNova (Optional) – Llama 3.1 405B",
                "Claude (Necessary) – Haiku",
                "Claude (Necessary) – Sonnet 3.5",
                "Google Gemini (Optional) – 1.5 Flash",
                "Blaxel (Optional) – MCP Model",
                "Local (Optional) – Tiny Model"
            ],
            value="Claude (Necessary) – Haiku",
            interactive=True
        )

        use_blaxel = gr.Checkbox(
            label="Enable Blaxel Suggestions (Sponsor)",
            value=False,
            info="Get additional insights and fun facts powered by Blaxel."
        )

        with gr.Row():
            user_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., Which animals live in the desert? or Tell me about the Golden Retriever."
            )
            
        submit_btn = gr.Button("Ask AI", variant="primary")
        
        gr.Markdown("### 🤖 AI Answer")
        output_box = gr.Markdown(value="*Ask a question to see the answer here...*")
        
        gr.Examples(
            examples=[
                ["Claude (Necessary) – Haiku", "Tell me about the tiger.", True],
                ["Claude (Necessary) – Haiku", "Which animals are canine?", True],
                ["Claude (Necessary) – Haiku", "Is a dolphin a cat?", False],
                ["SambaNova (Optional) – Llama 3.1 70B", "rare feline", True]
            ],
            inputs=[model_provider, user_input, use_blaxel],
            label="Example Queries"
        )
        
        # Gradio supports async functions natively
        submit_btn.click(
            fn=run_model,
            inputs=[model_provider, user_input, use_blaxel],
            outputs=output_box
        )
        
        gr.Markdown("---")
        # We can't easily show len(ANIMALS_DATA) here dynamically without state, 
        # but we can show a static message or load it
        gr.Markdown(f"**Status:** Server Ready.")
    return demo

# --- FastAPI Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cached_df
    try:
        # Initialize Semantic Search Model
        initialize_retriever()

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
        
        # Load data for Logic (List of Dicts)
        animals_list = load_data_list()
        set_animals_data(animals_list)
        
    except Exception as e:
        _cached_df = None
        set_animals_data([])
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
    set_animals_data([])

# --- Main App ---
app = FastAPI(title="Astrx MCP Server", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status":"UP"}

@app.get("/api/data")
def data_preview(limit: int = 100):
    if _cached_df is None:
        raise HTTPException(status_code=404, detail="No data available on the server")
    return _cached_df.head(limit).where(pd.notnull(_cached_df), None).to_dict(orient="records")

# Mount Gradio App
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
