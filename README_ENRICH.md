# Dataset Enrichment

To enrich the `data/animals.json` dataset with missing fields (diet, lifespan, fun_fact, threat_status), use the provided Modal script.

## Prerequisites

1.  **Modal**: Ensure you have Modal installed and authenticated (`modal setup`).
2.  **SambaNova API Key**: You need your SambaNova API key.

## Running the Enrichment

Run the following command in your terminal:

```bash
export SAMBANOVA_API_KEY="your_api_key_here"
modal run scripts/enrich_animals_modal.py
```

This script will:
1.  Load `data/animals.json`.
2.  Distribute the enrichment task across multiple Modal containers in parallel.
3.  Use the `Meta-Llama-3.1-8B-Instruct` model on SambaNova for fast processing.
4.  Save the enriched dataset to `data/animals_enriched.json`.

## Output

The output file `data/animals_enriched.json` will contain the same animals but with the following fields filled:
- `diet`
- `lifespan`
- `fun_fact`
- `threat_status`
- `source`: "AI-enriched"
