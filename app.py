from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import os
import gradio as gr
import json
from dotenv import load_dotenv

# Import logic
from logic import (
    run_model,
    set_animals_data,
    search_animals,
    initialize_retriever,
    save_keys,
    get_random_animal_fact,
    DEFAULT_SAMBANOVA_MODEL,
)

# Load environment variables
load_dotenv()

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_FILE = DATA_DIR / "animals_enriched.json"

# --- Global State ---
_cached_df = None

# --- Helper Functions ---
def load_data_list():
    """Loads and merges all animal datasets from the data directory."""
    all_data = []
    
    # 1. Load JSON files
    json_files = ["animals_enriched.json", "animals.json"]
    for jf in json_files:
        path = DATA_DIR / jf
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"Loaded {len(data)} records from {jf}")
                        all_data.extend(data)
                    else:
                        print(f"Warning: {jf} content is not a list.")
            except Exception as e:
                print(f"Error loading {jf}: {e}")

    # 2. Load CSV files
    csv_files = ["animals-external.csv", "Animal Traits Observations.csv"]
    for cf in csv_files:
        path = DATA_DIR / cf
        if path.exists():
            try:
                # Try loading with utf-8 first, then fallback to latin-1
                try:
                    df = pd.read_csv(path, encoding="utf-8")
                except UnicodeDecodeError:
                    print(f"Warning: {cf} is not UTF-8. Retrying with latin-1.")
                    df = pd.read_csv(path, encoding="latin-1")
                
                # Convert to list of dicts
                records = df.where(pd.notnull(df), None).to_dict(orient="records")
                print(f"Loaded {len(records)} records from {cf}")
                all_data.extend(records)
            except Exception as e:
                print(f"Error loading {cf}: {e}")
    
    print(f"Total records loaded: {len(all_data)}")
    return all_data

# --- Gradio UI ---
def create_gradio_app():
    with gr.Blocks(title="Astrx MCP Server") as demo:
        gr.Markdown("# 🚀 Astrx MCP Server")
        
        # --- Sponsor API Keys Panel ---
        with gr.Accordion("🔑 API Keys (Session Only)", open=True):
            gr.Markdown(
                "Enter your API keys below. They are stored in memory for this session only and are **not** saved to disk.\n\n"
                "**Note:** If you leave a field blank, the system will attempt to use a **Demo Key** (if available). "
                "Demo keys have strict rate limits and may fail if overused."
            )
            with gr.Row():
                samba_key = gr.Textbox(label="SambaNova Cloud API Key (Optional - Demo Available)", type="password", placeholder="Leave blank to use Demo Key")
                claude_key = gr.Textbox(label="Anthropic Claude API Key (Optional - Demo Available)", type="password", placeholder="Leave blank to use Demo Key")
            with gr.Row():
                modal_key = gr.Textbox(label="Modal Token (ID:Secret) (Optional)", type="password", placeholder="e.g., ak-123:as-456")
                blaxel_key = gr.Textbox(label="Blaxel API Key (Optional - Demo Available)", type="password", placeholder="Leave blank to use Demo Key")
            with gr.Row():
                gemini_key = gr.Textbox(label="Google Gemini API Key (Optional - Demo Available)", type="password", placeholder="Leave blank to use Demo Key")
            
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
                "SambaNova – Llama 3.3 70B",
                "SambaNova – Llama 3.1 8B",
                "SambaNova – DeepSeek R1",
                "SambaNova – DeepSeek R1 Distill",
                "SambaNova – DeepSeek V3",
                "SambaNova – DeepSeek V3.1",
                # Google - Frontier & Reasoning
                "Google – Gemini 3.0 Pro Preview",
                "Google – Gemini 2.5 Pro",
                # Google - High-Speed
                "Google – Gemini 2.5 Flash",
                "Google – Gemini 2.5 Flash-Lite",
                # Google - Specialized
                "Google – Gemini 2.5 Flash Image",
                # Google - Open Models
                "Google – Gemma 3",
                "Google – Gemma 3n",
                "Google – Gemma 2 27B",
                "Google – Gemma 2 9B",
                # Others
                "Blaxel – MCP Model",
                "Local – Tiny Model"
            ],
            value="SambaNova – Llama 3.3 70B",
            interactive=True
        )
        
        gr.Markdown(
            "⚠️ **Note:** Some advanced models require your own API key. "
            "If the response fails or hits rate limits, please add your key in the section above."
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
            
        with gr.Row():
            submit_btn = gr.Button("Ask AI", variant="primary")
            random_fact_btn = gr.Button("🎲 Random Animal Fact", variant="secondary")
        
        gr.Markdown("### 🤖 AI Answer")
        output_box = gr.Markdown(value="*Ask a question to see the answer here...*")
        
        # --- Event Handlers ---
        submit_btn.click(
            fn=run_model,
            inputs=[model_provider, user_input, use_blaxel],
            outputs=output_box
        )
        
        random_fact_btn.click(
            fn=get_random_animal_fact,
            inputs=[model_provider],
            outputs=output_box
        )
        
        gr.Examples(
            examples=[
                ["SambaNova – Llama 3.3 70B", "Tell me about the tiger.", True],
                ["Google – Gemini 2.5 Flash", "Which animals are canine?", True],
                ["Blaxel – MCP Model", "Is a dolphin a cat?", False],
                ["SambaNova – DeepSeek R1", "rare feline", True]
            ],
            inputs=[model_provider, user_input, use_blaxel],
            label="Example Queries"
        )
        
        gr.Markdown("---")
        gr.Markdown(
            "### 🏆 Hackathon Sponsors\n"
            "- **SambaNova Cloud**: Fast inference for Llama 3.1 & 3.3 models.\n"
            "- **Anthropic Claude**: High-intelligence reasoning.\n"
            "- **Blaxel**: Serverless agent hosting & suggestions.\n"
            "- **Modal**: Serverless GPU embeddings.\n\n"
            "*(Google Gemini is also supported as a model provider)*"
        )
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
                    try:
                        _cached_df = pd.read_csv(f, encoding="utf-8")
                    except UnicodeDecodeError:
                        _cached_df = pd.read_csv(f, encoding="latin-1")
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
