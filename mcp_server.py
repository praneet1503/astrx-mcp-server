import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, List

HAS_MCP = True
try:
    import mcp
    types = getattr(mcp, 'types', None)
    from mcp.server import NotificationOptions, Server, InitializationOptions
    from mcp.server.stdio import stdio_server
except Exception:
    HAS_MCP = False

# Import logic from app.py
# Ensure the current directory is in the path
sys.path.append(str(Path(__file__).parent))
from app import load_data, run_scraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP Server
if HAS_MCP:
    server = Server("astrx-mcp-server")
else:
    server = None

if HAS_MCP:
    @server.list_resources()
    async def handle_list_resources() -> list:
        """List available resources."""
        return [
            {
                "uri": "animals://data",
                "name": "Animal Dataset",
                "description": "The scraped dataset of animals",
                "mimeType": "application/json",
            }
        ]

    @server.read_resource()
    async def handle_read_resource(uri) -> str | bytes:
        """Read a specific resource."""
        # Some server may provide uri as a string; normalize
        uri_str = str(uri)
        if uri_str.startswith("animals://data") or uri_str == "animals://data":
            data = load_data()
            import json
            return json.dumps(data, indent=2)
        raise ValueError(f"Resource not found: {uri}")

    @server.list_tools()
    async def handle_list_tools() -> list:
        """List available tools."""
        return [
            {
                "name": "scrape_animals",
                "description": "Trigger the scraper to fetch fresh animal data",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list:
        """Handle tool execution."""
        if name == "scrape_animals":
            try:
                # We can call the logic from app.py
                # Note: app.scrape_data is an endpoint handler; we prefer running the helper directly.
                result = run_scraper()
                # Compose a simplified text result
                return [{"type": "text", "text": str(result)}]
            except Exception as e:
                return [{"type": "text", "text": f"Error scraping data: {e}"}]
            # Unknown tool
            raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout
    if not HAS_MCP:
        logger.error("MCP package not available in this environment. Exiting MCP server.")
        return
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
