import gradio as gr
import uvicorn
import json
import logging
import os
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

# Load configuration
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"logging_level": "INFO"}

# Configure logging
logging.basicConfig(level=config.get("logging_level", "INFO"))
logger = logging.getLogger(__name__)

# Define the tool
def list_animals() -> list[dict]:
    """
    Returns a list of animals with their details.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "animals.json")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading data file: {e}")
        return [{"name": "Error", "description": str(e)}]

# Initialize FastMCP server
mcp = FastMCP("Animal Service")
mcp.tool()(list_animals)

# Create main FastAPI app
app = FastAPI()

# Mount MCP app at /mcp to avoid conflicts
# Endpoints will be /mcp/sse and /mcp/messages
mcp_app = mcp.sse_app()
app.mount("/mcp", mcp_app)

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
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
