import json
import os
import httpx
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-haiku-20240307"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Global State ---
ANIMALS_DATA: List[Dict[str, Any]] = []
ANIMAL_EMBEDDINGS = None
RETRIEVER_MODEL = None

# --- Session Keys ---
SESSION_KEYS = {
    "samba": None,
    "claude": None,
    "modal": None,
    "blaxel": None
}

def save_keys(samba_key: str, claude_key: str, modal_key: str, blaxel_key: str) -> str:
    """
    Saves the provided API keys to the session storage.
    """
    global SESSION_KEYS
    SESSION_KEYS["samba"] = samba_key if samba_key and samba_key.strip() else None
    SESSION_KEYS["claude"] = claude_key if claude_key and claude_key.strip() else None
    SESSION_KEYS["modal"] = modal_key if modal_key and modal_key.strip() else None
    SESSION_KEYS["blaxel"] = blaxel_key if blaxel_key and blaxel_key.strip() else None
    
    return "Keys Saved Securely (Session Only)"

def validate_keys() -> Dict[str, bool]:
    """
    Returns a dictionary indicating which keys are present.
    """
    return {k: v is not None for k, v in SESSION_KEYS.items()}

def get_samba_key():
    """
    Returns the SambaNova API key if present.
    """
    key = SESSION_KEYS.get("samba") or os.getenv("SAMBANOVA_API_KEY")
    if not key:
        raise ValueError("SambaNova API Key is missing.")
    return key

def get_claude_key():
    """
    Returns the Claude API key if present.
    """
    key = SESSION_KEYS.get("claude") or os.getenv("CLAUDE_API_KEY")
    if not key:
        raise ValueError("Claude API Key is missing.")
    return key

def get_blaxel_key():
    """
    Returns the Blaxel API key if present.
    """
    key = SESSION_KEYS.get("blaxel")
    if not key:
        raise ValueError("Blaxel API Key is missing.")
    return key

def initialize_retriever():
    """
    Initializes the SentenceTransformer model.
    This should be called on startup.
    """
    global RETRIEVER_MODEL
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        RETRIEVER_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        RETRIEVER_MODEL = None

def set_animals_data(data: List[Dict[str, Any]]):
    """
    Sets the animal data and pre-computes embeddings.
    """
    global ANIMALS_DATA, ANIMAL_EMBEDDINGS
    ANIMALS_DATA = data
    
    if not RETRIEVER_MODEL:
        print("Warning: Retriever model not initialized. Semantic search will be disabled.")
        return

    if not data:
        ANIMAL_EMBEDDINGS = None
        return

    print(f"Computing embeddings for {len(data)} records...")
    # Create a text representation for each animal for embedding
    # Format: "Name: [name]. Description: [description]"
    texts = [
        f"Name: {animal.get('name', '')}. Description: {animal.get('description', '')}" 
        for animal in data
    ]
    
    try:
        ANIMAL_EMBEDDINGS = RETRIEVER_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        print("Embeddings computed.")
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        ANIMAL_EMBEDDINGS = None

def search_animals(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Semantic search using vector embeddings.
    Falls back to keyword search if model is not available.
    """
    if not query:
        return []

    # Fallback to keyword search if semantic search is not ready
    if RETRIEVER_MODEL is None or ANIMAL_EMBEDDINGS is None:
        print("Using fallback keyword search.")
        query_words = query.lower().split()
        matches = []
        for animal in ANIMALS_DATA:
            text = (animal.get("name", "") + " " + str(animal.get("description", ""))).lower()
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                matches.append((score, animal))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:top_k]]

    # Semantic Search
    try:
        # Encode query
        query_embedding = RETRIEVER_MODEL.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
        # Calculate Cosine Similarity
        # (1, D) @ (N, D).T -> (1, N)
        similarities = np.dot(ANIMAL_EMBEDDINGS, query_embedding)
        
        # Get top-k indices
        # argsort returns ascending, so we take the last k and reverse
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.2: 
                # Create a copy to avoid mutating the global list
                item = ANIMALS_DATA[idx].copy()
                item['_score'] = score
                results.append(item)
                
        return results
        
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []

# --- Model Runners (Stubs) ---

async def run_samba(prompt: str, model_version: str) -> str:
    """
    Executes the prompt using SambaNova's API via httpx.
    """
    try:
        api_key = get_samba_key()
        
        # Map UI names to actual model IDs
        model = "Meta-Llama-3.1-8B-Instruct"
        if "405B" in model_version:
            model = "Meta-Llama-3.1-405B-Instruct"
        elif "70B" in model_version:
            model = "Meta-Llama-3.1-70B-Instruct"
            
        url = "https://api.sambanova.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "top_p": 0.1
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
    except ValueError as e:
        return f"Configuration Error: {str(e)}"
    except httpx.HTTPStatusError as e:
        return f"SambaNova API Error: {e.response.text}"
    except Exception as e:
        return f"SambaNova Error: {str(e)}"

async def run_claude(prompt: str, model_version: str) -> str:
    """
    Executes the prompt using Anthropic's Claude API via httpx.
    Raises exceptions on failure so the caller can handle fallbacks.
    """
    # 1. Get Key (will raise ValueError if missing)
    api_key = get_claude_key()
    
    # 2. Map Model ID
    model = "claude-3-haiku-20240307"
    if "Sonnet" in model_version:
        model = "claude-3-5-sonnet-20240620"
        
    # 3. Prepare Request
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    # 4. Execute Async Call
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        text = response.json()["content"][0]["text"]
        
    # 5. Format Output
    return f"**Powered by Claude**\n\n{text}"

async def run_blaxel(prompt: str) -> str:
    """
    Executes the prompt using Blaxel's API via httpx.
    """
    try:
        api_key = get_blaxel_key()
        
        url = "https://api.blaxel.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "blaxel-mcp",
            "prompt": prompt
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["output"]
            
    except ValueError as e:
        return f"Configuration Error: {str(e)}"
    except httpx.HTTPStatusError as e:
        return f"Blaxel API Error: {e.response.text}"
    except Exception as e:
        return f"Blaxel Error: {str(e)}"

def run_local(prompt: str) -> str:
    """
    Stub for Local inference.
    """
    return f"[Local Dummy Model] You said: {prompt[:200]}"

async def run_model(provider: str, user_input: str) -> str:
    """
    Unified routing function to dispatch requests to the selected provider.
    """
    if not user_input or not user_input.strip():
        return "Please enter a valid question."

    # 1. Retrieve Context (RAG)
    if not ANIMALS_DATA:
        return "Error: No animal data loaded."

    relevant_animals = search_animals(user_input)
    if not relevant_animals:
        context_str = "No specific animals found matching the query."
    else:
        context_str = json.dumps(relevant_animals, ensure_ascii=False)

    # 2. Construct Prompt with Context
    full_prompt = (
        f"Context:\n{context_str}\n\n"
        f"User Question: {user_input}\n\n"
        "Answer based on the context provided."
    )

    # 3. Route to Provider
    try:
        if provider.startswith("SambaNova"):
            # Extract version if needed, e.g., "SambaNova – Samba-1" -> "Samba-1"
            version = provider.split("–")[-1].strip()
            return await run_samba(full_prompt, version)
        
        elif provider.startswith("Claude"):
            version = provider.split("–")[-1].strip()
            try:
                return await run_claude(full_prompt, version)
            except Exception as e:
                # Fallback for Claude
                print(f"Claude API failed: {e}")
                return (
                    f"**Powered by Claude** (Unavailable)\n\n"
                    f"Claude is unavailable: {str(e)}\n\n"
                    f"Showing keyword search result instead:\n\n{context_str}"
                )
        
        elif provider.startswith("Blaxel"):
            return await run_blaxel(full_prompt)
        
        elif provider.startswith("Local"):
            return run_local(full_prompt)
        
        else:
            return f"Error: Unknown provider '{provider}'"
            
    except Exception as e:
        return f"Routing Error: {str(e)}"

# Deprecated: Kept for backward compatibility if needed, but run_model should be used.
async def query_claude(user_input: str) -> str:
    return await run_model("Claude – Haiku", user_input)

