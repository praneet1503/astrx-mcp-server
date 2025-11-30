import modal
import json
import os
import asyncio
import httpx
import argparse
from dotenv import load_dotenv

# Load local .env file if present
load_dotenv()

# Define the Modal App
app = modal.App("enrich-animals")

# Define the image with dependencies
image = modal.Image.debian_slim().pip_install("httpx", "python-dotenv")

import time
import random

# Define the secret for API keys
secrets = []

# 1. Try to get from local environment
if "SAMBANOVA_API_KEY" in os.environ:
    secrets.append(modal.Secret.from_local_environ(["SAMBANOVA_API_KEY"]))

# 2. Try to load from .env file
secrets.append(modal.Secret.from_dotenv())

@app.function(image=image, secrets=secrets, timeout=300, max_containers=2)
async def enrich_animal(animal: dict):
    """
    Enriches a single animal record with missing fields using SambaNova API.
    """
    # Check if already enriched
    if animal.get("diet") and animal.get("lifespan") and animal.get("fun_fact") and animal.get("threat_status"):
        return animal

    api_key = os.environ.get("SAMBANOVA_API_KEY")
    if not api_key:
        print(f"Skipping {animal.get('name')}: SAMBANOVA_API_KEY not found.")
        return animal

    # Add a small delay to avoid hitting rate limits immediately
    await asyncio.sleep(1.0)

    model_id = "Meta-Llama-3.3-70B-Instruct"
    
    prompt = f"""
    You are an expert zoologist. 
    For the animal "{animal.get('name')}", provide the following missing details:
    - diet (concise, e.g. "Carnivore, eats deer")
    - lifespan (concise, e.g. "10-15 years")
    - fun_fact (one interesting fact)
    - threat_status (e.g. Endangered, Vulnerable, Least Concern)

    Return ONLY a valid JSON object with these 4 keys. Do not include markdown formatting or explanations.
    Example: {{"diet": "Carnivore", "lifespan": "10-15 years", "fun_fact": "...", "threat_status": "Vulnerable"}}
    """

    url = "https://api.sambanova.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 150
    }

    max_retries = 10
    base_delay = 5

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 429:
                    print(f"429 Response Body: {response.text}")
                    delay = base_delay * (1.5 ** attempt) + random.uniform(1, 3)
                    print(f"Rate limited for {animal.get('name')}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                
                response.raise_for_status()
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Clean up content if it has markdown code blocks
                content = content.replace("```json", "").replace("```", "").strip()
                
                enriched_data = json.loads(content)
                
                # Update fields if they are missing in original
                if not animal.get("diet"):
                    animal["diet"] = enriched_data.get("diet", "")
                if not animal.get("lifespan"):
                    animal["lifespan"] = enriched_data.get("lifespan", "")
                if not animal.get("fun_fact"):
                    animal["fun_fact"] = enriched_data.get("fun_fact", "")
                if not animal.get("threat_status"):
                    animal["threat_status"] = enriched_data.get("threat_status", "")
                
                animal["source"] = "AI-enriched"
                return animal
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to enrich {animal.get('name')} after {max_retries} attempts: {e}")
                return animal
            await asyncio.sleep(1)
    
    return animal

@app.local_entrypoint()
def main(limit: int = 0, test: bool = False):
    input_file = "data/animals.json"
    output_file = "data/animals_enriched.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r") as f:
        animals = json.load(f)
    
    print(f"Loaded {len(animals)} animals from {input_file}.")
    
    # Check for API Key locally
    if not os.environ.get("SAMBANOVA_API_KEY"):
        print("WARNING: SAMBANOVA_API_KEY is not set in your local environment.")
        print("Attempting to rely on Modal Secrets from .env...")

    # Determine how many to process
    if test:
        print("🧪 TEST MODE: Processing only 10 animals.")
        animals_to_process = animals[:10]
    elif limit > 0:
        print(f"⚠️ LIMIT MODE: Processing first {limit} animals.")
        animals_to_process = animals[:limit]
    else:
        print("🚀 FULL MODE: Processing ALL animals.")
        animals_to_process = animals

    enriched_animals = []
    count = 0
    
    # Process in parallel using Modal map
    print(f"Starting enrichment process on Modal for {len(animals_to_process)} animals...")
    try:
        for result in enrich_animal.map(animals_to_process):
            enriched_animals.append(result)
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{len(animals_to_process)} animals...")
    except Exception as e:
        print(f"Error during processing: {e}")
        # Save whatever we have
        with open(output_file, "w") as f:
            json.dump(enriched_animals, f, indent=2)
        print(f"Saved partial results to {output_file}")
        return

    # If we only processed a subset, we should probably merge with the original list
    # to avoid losing data.
    if len(enriched_animals) < len(animals):
        print("Merging enriched subset with original dataset...")
        # Create a map of enriched animals by name
        enriched_map = {a['name']: a for a in enriched_animals}
        
        final_list = []
        for original in animals:
            if original['name'] in enriched_map:
                final_list.append(enriched_map[original['name']])
            else:
                final_list.append(original)
        
        with open(output_file, "w") as f:
            json.dump(final_list, f, indent=2)
    else:
        with open(output_file, "w") as f:
            json.dump(enriched_animals, f, indent=2)
    
    print(f"Success! Enriched {len(enriched_animals)} animals. Saved to {output_file}.")

if __name__ == "__main__":
    # This allows running locally if needed, but Modal requires 'modal run'
    pass
