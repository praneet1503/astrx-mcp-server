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
  - google
  - gemini
  - anthropic
  - modal
  - animals
---

# 🦁 Astrx MCP Server

**The Ultimate Multi-Model RAG System for Animal Knowledge.**

**Astrx MCP Server** is a high-performance, production-ready Model Context Protocol (MCP) server designed to provide accurate, scientifically backed, and AI-enriched information about the animal kingdom. It combines a massive local dataset (19,000+ records) with advanced Large Language Models (LLMs) to deliver instant, reliable answers.

🎥 **[Watch the Demo Video](https://drive.google.com/file/d/1OYoQpFaaLm8U5TRkCWrPp7kfGbTltG2p/view?usp=drive_link)**

---

## 🚀 Features

### 🧠 Multi-Model Intelligence
Seamlessly switch between top-tier AI models for reasoning and generation:
*   **SambaNova Cloud**: Blazing fast inference with Llama 3.3 70B, Llama 3.1 8B, and DeepSeek R1/V3.
*   **Google Gemini**: Full support for the Gemini ecosystem (3.0 Pro Preview, 2.5 Pro, 2.5 Flash, Gemma 3).
*   **Anthropic Claude**: High-intelligence reasoning with Claude 3.5 Sonnet and Haiku.
*   **Blaxel**: Serverless agent suggestions and fun facts.
*   **Local Fallback**: Offline keyword search and "Tiny Model" simulation.

### 🛡️ Robust API Key System & Failover
*   **Optional Keys**: Users can explore the system using built-in **Demo Keys** (subject to rate limits).
*   **Smart Fallback**: If a premium model (e.g., Gemini 2.5 Pro) hits rate limits, the system automatically downgrades to a faster, cheaper model (e.g., Gemini 2.5 Flash-Lite) to ensure the request succeeds.
*   **Session Security**: User-provided API keys are stored in memory only for the active session and never saved to disk.

### 📚 Unified Knowledge Base
The system queries a consolidated index built from multiple sources:
1.  **Scraped Data**: 19,000+ raw animal records (`animals.json`).
2.  **Scientific Data**: Taxonomy and classification from external CSVs (`animals-external.csv`).
3.  **AI-Enriched Data**: Missing fields (diet, lifespan, fun facts) filled by Llama 3.3 via Modal (`animals_enriched.json`).

### 🔍 Hybrid Search (RAG)
*   **Semantic Search**: Uses `sentence-transformers` (Local) or Modal (Serverless GPU) to find animals based on meaning, not just keywords.
*   **Keyword Fallback**: Instant results even without a GPU.

---

## 🧩 Architecture

The project is structured as a modular **FastAPI** application with a **Gradio** frontend.

*   **Frontend (`app.py`)**: A clean, interactive web interface for querying models, managing keys, and viewing raw data.
*   **Logic Core (`logic.py`)**: Handles RAG retrieval, prompt engineering, model routing, and error handling.
*   **Data Layer (`data/`)**: Stores JSON and CSV datasets.
*   **ETL Pipelines (`scripts/`)**: Python scripts for merging, cleaning, and enriching data.

---

## 📂 Folder Structure

```plaintext
astrx-mcp-server/
├── app.py                      # Main entry point (FastAPI + Gradio)
├── logic.py                    # Core business logic, RAG, and model integration
├── modal_ops.py                # Modal serverless functions for embeddings
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── README.md                   # Documentation
├── data/                       # Dataset storage
│   ├── animals.json            # Raw scraped data
│   ├── animals_enriched.json   # AI-processed production data
│   ├── animals-external.csv    # Scientific classification data
│   └── Animal Traits...csv     # Observational traits
├── scripts/                    # ETL and Maintenance
│   ├── merge_external_datasets.py
│   ├── enrich_animals_modal.py
│   └── validate_data.py
└── tests/                      # Unit and pipeline tests
```

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.10+
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/praneet1503/astrx-mcp-server.git
cd astrx-mcp-server
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)
Create a `.env` file to store your API keys permanently for local development:
```ini
SAMBANOVA_API_KEY=your_key_here
CLAUDE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
MODAL_TOKEN_ID=your_token_id
MODAL_TOKEN_SECRET=your_token_secret
```

### 4. Run the Server
```bash
python app.py
```
Access the UI at `http://localhost:7860`.

---

## 🧬 Dataset & Enrichment

The heart of Astrx is its data. We use a multi-stage pipeline to ensure quality:

1.  **Ingestion**: Raw data is scraped and saved to `data/animals.json`.
2.  **Normalization**: `scripts/merge_external_datasets.py` standardizes field names and merges scientific classification.
3.  **AI Enrichment**: `scripts/enrich_animals_modal.py` uses **Modal** to parallelize requests to **SambaNova Llama 3.1 8B**, filling in gaps like:
    *   *Diet*
    *   *Lifespan*
    *   *Threat Status*
    *   *Fun Facts*

To run the enrichment pipeline yourself:
```bash
export SAMBANOVA_API_KEY="your_key"
modal run scripts/enrich_animals_modal.py
```

---

## ☁️ Deployment

### Hugging Face Spaces
This project is optimized for Hugging Face Spaces (Docker SDK).
1.  Create a new Space.
2.  Select **Docker** as the SDK.
3.  Push this repository to the Space.
4.  Add your API keys as **Secrets** in the Space settings.

### Modal
Used for heavy lifting (embeddings and batch enrichment).
*   Deploy the embedding function: `modal deploy modal_ops.py`

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes.
4.  Push to the branch.
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

**Built with ❤️ for the MCP Hackathon.**
*Powered by SambaNova, Google Gemini, Anthropic, Blaxel, and Modal.*

