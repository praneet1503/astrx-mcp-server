---
title: Astrx Mcp Server
emoji: 📉
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'an mcp server to get all the details about the animals '
---

# Animal MCP Server

A minimal Model Context Protocol (MCP) server that serves animal data.

## Features

- **Tools**:
  - `list_animals`: Returns a list of animals with details (name, species, description, traits).
- **Transport**:
  - Supports **stdio** for Claude Desktop integration.
  - Supports **SSE** (Server-Sent Events) for local HTTP serving via `uvicorn`.

## Setup

1. **Install Dependencies**:
   Make sure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   - `config.json` allows setting the logging level.

## Running the Server

### Option 1: Local HTTP Server (SSE)
To run the server locally with `uvicorn` (useful for testing or HTTP clients):

```bash
uvicorn server:app --reload
```
The server will be available at `http://localhost:8000`.
- SSE Endpoint: `http://localhost:8000/sse`
- Messages Endpoint: `http://localhost:8000/messages`

### Option 2: Claude Desktop (Stdio)
To connect this server to Claude Desktop:

1. Open your Claude Desktop configuration file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the following to the `mcpServers` object:

```json
{
  "mcpServers": {
    "animals": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/hf_mcp_server",
        "run",
        "server.py"
      ]
    }
  }
}
```

## Data Scraping

The project includes a modular scraping system to fetch animal data from multiple sources.

### Running the Scraper
The scraper runs on **Modal** to handle browser automation and anti-bot measures.

```bash
modal run scripts/fetch_animals.py
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
