import modal
import json
import os
import asyncio
import httpx

# Define the Modal App
app = modal.App("enrich-animals")

# Define the image with dependencies
image = modal.Image.debian_slim().pip_install("httpx")

# Define the secret for API keys
# We expect SAMBANOVA_API_KEY to be in the local environment when running the script
secrets = [modal.Secret.from_local_environ(["SAMBANOVA_API_KEY"])]

@app.function(image=image, secrets=secrets, timeout=60, concurrency_limit=50)
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

    model_id = "Meta-Llama-3.1-8B-Instruct"
    
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

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
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
        print(f"Failed to enrich {animal.get('name')}: {e}")
        return animal

@app.local_entrypoint()
def main():
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
        print("Please run: export SAMBANOVA_API_KEY='your_key' before running this script.")
        return

    enriched_animals = []
    count = 0
    
    # Process in parallel using Modal map
    print("Starting enrichment process on Modal...")
    try:
        for result in enrich_animal.map(animals):
            enriched_animals.append(result)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{len(animals)} animals...")
    except Exception as e:
        print(f"Error during processing: {e}")
        # Save whatever we have
        with open(output_file, "w") as f:
            json.dump(enriched_animals, f, indent=2)
        print(f"Saved partial results to {output_file}")
        return

    with open(output_file, "w") as f:
        json.dump(enriched_animals, f, indent=2)
    
    print(f"Success! Enriched {len(enriched_animals)} animals. Saved to {output_file}.")

if __name__ == "__main__":
    # This allows running locally if needed, but Modal requires 'modal run'
    pass
