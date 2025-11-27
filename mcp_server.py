import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, List

import mcp.types as types
from mcp.server import NotificationOptions, Server, InitializationOptions
from mcp.server.stdio import stdio_server

# Import logic from app.py
# Ensure the current directory is in the path
sys.path.append(str(Path(__file__).parent))
from app import load_dataset, scrape_data as app_scrape_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP Server
server = Server("astrx-mcp-server")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available resources.
    """
    return [
        types.Resource(
            uri=types.AnyUrl("animals://data"),
            name="Animal Dataset",
            description="The scraped dataset of animals",
            mimeType="application/json",
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: types.AnyUrl) -> str | bytes:
    """
    Read a specific resource.
    """
    if uri.scheme == "animals" and uri.path == "/data":
        data = load_dataset()
        import json
        return json.dumps(data, indent=2)
    raise ValueError(f"Resource not found: {uri}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    """
    return [
        types.Tool(
            name="scrape_animals",
            description="Trigger the scraper to fetch fresh animal data",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution.
    """
    if name == "scrape_animals":
        try:
            # We can call the logic from app.py
            # Note: app.scrape_data is an endpoint handler, it might return a dict or raise HTTPException
            # We should probably wrap it or call the underlying logic.
            # app.scrape_data calls scraper.main() and handles global cache.
            
            # Let's just call scraper directly to avoid FastAPI dependency issues if any,
            # but reusing app logic ensures consistency with the cache.
            # However, app.scrape_data is synchronous.
            
            # We'll run it in a thread to not block the async loop if it takes time
            # But for simplicity, let's just call it.
            
            # We need to handle the fact that app.scrape_data might raise HTTPException
            try:
                result = app_scrape_data()
                return [types.TextContent(type="text", text=str(result))]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error scraping data: {str(e)}")]
                
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="astrx-mcp-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
