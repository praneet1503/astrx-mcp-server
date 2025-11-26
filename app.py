import gradio as gr
import uvicorn
import json
import logging
import os
import sys
from fastapi import FastAPI, Request
from mcp.server.fastmcp import FastMCP

# Load configuration
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"logging_level": "INFO"}

# Configure logging
logging.basicConfig(
    level=config.get("logging_level", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Define the tool
def list_animals() -> list[dict]:
    """
    Returns a list of animals with their details.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "animals.json")
    logger.info(f"Reading data from {data_path}")
    try:
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at {data_path}")
            return [{"name": "Error", "description": "Data file not found"}]
            
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} animals")
            return data
    except Exception as e:
        logger.error(f"Error reading data file: {e}", exc_info=True)
        return [{"name": "Error", "description": str(e)}]

# Initialize FastMCP server
logger.info("Initializing FastMCP server...")
mcp = FastMCP("Animal Service")
mcp.tool()(list_animals)

# Create main FastAPI app
app = FastAPI()

# Add middleware to log errors
@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        raise e

# Mount MCP app at /mcp to avoid conflicts
# Endpoints will be /mcp/sse and /mcp/messages
try:
    mcp_app = mcp.sse_app()
    app.mount("/mcp", mcp_app)
    logger.info("Mounted MCP app at /mcp")
except Exception as e:
    logger.error(f"Failed to mount MCP app: {e}", exc_info=True)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Animal MCP Server")
    gr.Markdown("## Status: Running")
    gr.Markdown("The MCP server is running and accessible at the following endpoints:")
    gr.Markdown("- **SSE Endpoint**: `/mcp/sse`")
    gr.Markdown("- **Messages Endpoint**: `/mcp/messages`")
    
    gr.Markdown("### Test Tool")
    output = gr.JSON(label="Animals List")
    btn = gr.Button("List Animals")
    btn.click(fn=list_animals, outputs=output)

# Mount Gradio app at root
logger.info("Mounting Gradio app...")
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
