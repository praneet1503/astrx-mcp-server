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

This project integrates external scientific datasets with a massive scraped database (19,000+ animals) to provide accurate, normalized, and AI-enriched answers.

## 🧬 External Data Integration

We combine raw scraped data with high-quality external sources to ensure accuracy:

1.  **Animalia Kingdom Classification**: Adds scientific taxonomy (Phylum, Class, Order, Family, Genus).
2.  **Animal Lifestyle & Traits**: Normalizes diet, habitat, and behavioral patterns.

## 🧩 Architecture

*   **`data/animals.json`**: Raw output from the parallel web scraper.
*   **`data/external/`**: Contains scientific CSV datasets (`classification.csv`, `traits.csv`).
*   **`scripts/merge_external_datasets.py`**: The ETL pipeline that:
    *   Normalizes field names (`scientific_name` → `name`).
    *   Merges external traits into the main dataset.
    *   Generates the production-ready `data/animals_enriched.json`.
*   **`scripts/enrich_animals_modal.py`**: Fallback LLM pipeline (SambaNova Llama 3.3) to fill any remaining gaps.
*   **`app.py`**: Gradio frontend serving the enriched dataset.

## 🛠️ How to Rebuild the Dataset

To regenerate `animals_enriched.json` locally:

1.  **Place Datasets**: Ensure `classification.csv` and `traits.csv` are in `data/external/`.
2.  **Run Merge Script**:
    ```bash
    python scripts/merge_external_datasets.py
    ```
3.  **Run AI Enrichment (Optional)**:
    If fields are still missing, use the Modal pipeline:
    ```bash
    modal run scripts/enrich_animals_modal.py
    ```

## 🚀 Usage

The Space automatically loads `data/animals_enriched.json`.
Simply enter your API key (SambaNova, Claude, or Gemini) and ask questions like *"What is the scientific classification of a Tiger?"* to see the merged data in action.

