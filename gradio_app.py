import gradio as gr
import json
import requests
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# The path to your dataset
DATA_FILE = "data/animals.json"
# Claude API Endpoint
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
# Model to use (Haiku is fast and cheap, Sonnet/Opus are smarter)
CLAUDE_MODEL = "claude-3-haiku-20240307" 

# --- 1. Load Data ---
def load_data():
    """Loads the animals dataset from the JSON file."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} records from {DATA_FILE}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

# Load data once at startup
ANIMALS_DATA = load_data()

# --- 2. Define the Search/Query Logic ---
def search_animals(query):
    """
    Simple keyword search to find relevant animals.
    Returns a list of animal dictionaries.
    """
    query_words = query.lower().split()
    matches = []
    
    for animal in ANIMALS_DATA:
        # Search in name and description
        text = (animal.get("name", "") + " " + str(animal.get("description", ""))).lower()
        
        # Score based on how many query words appear
        score = sum(1 for word in query_words if word in text)
        
        if score > 0:
            matches.append((score, animal))
            
    # Sort by score (descending) and take top 20
    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:20]]

def query_claude(user_input):
    """
    Sends the user input and the dataset context to Claude.
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        return "Error: CLAUDE_API_KEY not found in environment variables."
    
    if not ANIMALS_DATA:
        return "Error: No animal data loaded. Please check the server logs."

    # Retrieval Step: Find relevant animals to reduce token count
    relevant_animals = search_animals(user_input)
    
    if not relevant_animals:
        # Fallback: If no specific matches, maybe send a list of names or just say so
        # For now, we'll send a message saying we couldn't find specific data
        context_str = "No specific animals found matching the query in the dataset."
    else:
        context_str = json.dumps(relevant_animals, ensure_ascii=False)

    print(f"Sending {len(relevant_animals)} records to Claude...")

    system_prompt = (
        "You are an expert zoologist assistant. "
        "You have access to a subset of the animal dataset relevant to the user's query below. "
        "Your task is to answer the user's question based strictly on this provided data. "
        "If the answer is not in the data, say so. "
        "Cite the 'name' of the animal when providing facts.\n\n"
        f"RELEVANT DATA:\n{context_str}"
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    try:
        # --- 3. Send Request to Claude ---
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the text content
            content = result.get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "No text returned.")
            else:
                return "Empty response from Claude."
        else:
            return f"API Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"Request Failed: {str(e)}"

# --- 4. Build the Gradio UI ---
def create_app():
    with gr.Blocks(title="Astrx MCP Server") as app:
        gr.Markdown("# 🚀 Astrx MCP Server")
        gr.Markdown(
            "Ask questions about the animal dataset! "
            "This tool uses **Claude 3** to reason over the `animals.json` file."
        )
        
        with gr.Row():
            user_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., Which animals live in the desert? or Tell me about the Golden Retriever."
            )
            
        submit_btn = gr.Button("Ask Claude", variant="primary")
        
        output_box = gr.Textbox(label="Claude's Answer", lines=10)
        
        # Event handling
        submit_btn.click(
            fn=query_claude,
            inputs=[user_input],
            outputs=output_box
        )
        
        gr.Markdown("---")
        gr.Markdown(f"**Dataset Status:** Loaded {len(ANIMALS_DATA)} records.")

    return app

# --- 5. Launch ---
if __name__ == "__main__":
    # Create and launch the app
    demo = create_app()
    # server_name="0.0.0.0" is required for Hugging Face Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860)
