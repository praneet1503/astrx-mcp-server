from __future__ import annotations
import gradio as gr
import uvicorn
import json
import logging
import os
import sys
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration & Logging ---
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"logging_level": "INFO"}

logging.basicConfig(
    level=config.get("logging_level", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Global Data Cache ---
ANIMALS_CACHE: List[Dict[str, Any]] = []

def load_data():
    """Loads animals data into the global cache."""
    global ANIMALS_CACHE
    data_path = os.path.join(os.path.dirname(__file__), "data", "animals.json")
    try:
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                ANIMALS_CACHE = json.load(f)
            logger.info(f"Successfully loaded {len(ANIMALS_CACHE)} animals into cache.")
        else:
            logger.warning(f"Data file not found at {data_path}. Cache is empty.")
            ANIMALS_CACHE = []
    except Exception as e:
        logger.error(f"Error loading data file: {e}", exc_info=True)
        ANIMALS_CACHE = []

# --- Tool Definition ---
def list_animals() -> List[Dict[str, Any]]:
    """
    Returns a list of animals with their details from the in-memory cache.
    """
    # If cache is empty, try reloading (in case file was added later)
    if not ANIMALS_CACHE:
        load_data()
    
    if not ANIMALS_CACHE:
         return [{"name": "Error", "description": "No data available"}]
         
    return ANIMALS_CACHE

# --- MCP Server Setup (Conditional) ---
mcp = None
try:
    from mcp.server.fastmcp import FastMCP
    logger.info("Initializing FastMCP server...")
    mcp = FastMCP("Animal Service")
    mcp.tool()(list_animals)
except ImportError:
    logger.error("Failed to import FastMCP. MCP features will be disabled.")
except Exception as e:
    logger.error(f"Failed to initialize FastMCP: {e}", exc_info=True)

# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    load_data()
    yield

app = FastAPI(lifespan=lifespan)

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Spaces/Dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Error Handling Middleware
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "error": str(e)}
        )

# 3. Startup Event (Removed in favor of lifespan)
# @app.on_event("startup")
# async def startup_event():
#     load_data()

# 4. Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "mcp_enabled": mcp is not None,
        "animals_count": len(ANIMALS_CACHE)
    }

# 5. Mount MCP (if available)
if mcp:
    try:
        mcp_app = mcp.sse_app()
        app.mount("/mcp", mcp_app)
        logger.info("Mounted MCP app at /mcp")
    except Exception as e:
        logger.error(f"Failed to mount MCP app: {e}", exc_info=True)

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Animal MCP Server")
    gr.Markdown(f"## Status: {'Running' if mcp else 'Running (MCP Disabled)'}")
    
    if mcp:
        gr.Markdown("The MCP server is active at:")
        gr.Markdown("- **SSE Endpoint**: `/mcp/sse`")
        gr.Markdown("- **Messages Endpoint**: `/mcp/messages`")
    else:
        gr.Markdown("⚠️ **MCP Server is not active** (Import failed or initialization error).")

    gr.Markdown("### Test Tool")
    output = gr.JSON(label="Animals List")
    btn = gr.Button("List Animals")
    # Fix: outputs should be a list or single component, but explicit list is safer
    btn.click(fn=list_animals, outputs=[output])

# 6. Mount Gradio at Root
# This is critical for HF Spaces to render the UI correctly without 404s
logger.info("Mounting Gradio app at root / ...")
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Use PORT env var for HF Spaces, fallback to 7860
    port = int(os.getenv("PORT", "7860"))
    host = "0.0.0.0"
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
