---
title: Astrx MCP Server
emoji: 🦁
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
tags:
  - mcp
  - rag
  - sambanova
  - modal
  - animals
---

# 🦁 Astrx MCP Server

**A high-performance, multi-model RAG system for animal knowledge.**

This project combines a massive dataset of 19,000+ animals with cutting-edge LLMs to provide accurate, cited answers. It features a custom parallel web scraper and an AI-powered data enrichment pipeline.

## 🚀 Key Features

*   **⚡ SambaNova Cloud Integration**: Blazing fast inference using **Llama 3.3 70B**, **DeepSeek R1**, and **DeepSeek V3**.
*   **🕷️ Parallel Scraping Engine**: Built on **Modal** to scrape thousands of pages concurrently from A-Z Animals and other sources.
*   **🧠 Multi-Provider Support**: Seamlessly switch between **SambaNova**, **Anthropic (Claude)**, **Google (Gemini)**, and **Blaxel**.
*   **🔍 Hybrid RAG**: Combines semantic search (SentenceTransformers) with keyword fallback for precise retrieval.
*   **💎 AI-Enriched Data**: Dataset automatically enriched with diet, lifespan, and threat status using Llama 3.1.

## 🎮 How to Use

1.  **Enter API Key**: Input your **SambaNova API Key** (recommended for best performance) or keys for other providers.
2.  **Select Model**: Choose a specific model (e.g., `DeepSeek-R1-Distill-Llama-70B` or `Meta-Llama-3.3-70B`).
3.  **Ask Anything**: Query the system (e.g., *"What is the diet of a Snow Leopard?"*).
4.  **View Context**: See exactly which database entries were used to generate the answer.

## 🛠️ Environment Variables

To run locally or deploy, set these in your `.env` or Space secrets:

*   `SAMBANOVA_API_KEY`: Primary inference provider.
*   `CLAUDE_API_KEY`: For Anthropic models.
*   `GEMINI_API_KEY`: For Google models.
*   `BLAXEL_API_KEY`: For Blaxel serverless agents.
*   `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`: Required for running the scraper or remote embeddings.

## 🧩 Architecture

*   **Frontend**: Built with **Gradio** for a responsive chat interface.
*   **Logic Core (`logic.py`)**: Handles request routing, RAG retrieval, and prompt engineering.
*   **Data Pipeline**:
    *   `local_scraper/`: Async Playwright scraper running on Modal.
    *   `scripts/enrich_animals_modal.py`: Massively parallel data enrichment script.
    *   `data/animals.json`: The source of truth (19k+ records).

## 👩‍💻 Developer Usage

Run the RAG logic programmatically:

```python
from logic import search_animals, run_samba

# 1. Retrieve Context
context_docs = await search_animals("Tiger diet", top_k=3)

# 2. Generate Answer
context_str = "\n".join([str(d) for d in context_docs])
prompt = f"Context: {context_str}\n\nQuestion: What do tigers eat?"
response = await run_samba(prompt, model_choice="Meta-Llama-3.3-70B-Instruct")

print(response)
```

