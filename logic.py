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

def get_samba_client():
    """
    Returns a SambaNova client if the key is present.
    """
    key = SESSION_KEYS.get("samba")
    if not key:
        raise ValueError("SambaNova API Key is missing.")
    # Placeholder for actual SDK initialization
    # from openai import OpenAI
    # return OpenAI(api_key=key, base_url="https://api.sambanova.ai/v1")
    return {"client": "samba_placeholder", "key_present": True}

def get_claude_client():
    """
    Returns an Anthropic client if the key is present.
    """
    key = SESSION_KEYS.get("claude") or os.getenv("CLAUDE_API_KEY")
    if not key:
        raise ValueError("Claude API Key is missing.")
    # Placeholder or actual SDK
    # import anthropic
    # return anthropic.AsyncAnthropic(api_key=key)
    return {"client": "claude_placeholder", "key_present": True, "key": key}

def get_modal_client():
    """
    Returns a Modal client if the key is present.
    """
    key = SESSION_KEYS.get("modal")
    if not key:
        raise ValueError("Modal API Token is missing.")
    # Modal usually uses config/env vars, but we can return the token for manual usage
    return {"client": "modal_placeholder", "key_present": True}

def get_blaxel_client():
    """
    Returns a Blaxel client if the key is present.
    """
    key = SESSION_KEYS.get("blaxel")
    if not key:
        raise ValueError("Blaxel API Key is missing.")
    return {"client": "blaxel_placeholder", "key_present": True}

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

async def query_claude(user_input: str) -> str:
    """
    Sends the user input and the dataset context to Claude asynchronously.
    """
    if not user_input or not user_input.strip():
        return "Please enter a valid question."

    # Try to get key from session first, then env var
    api_key = SESSION_KEYS.get("claude") or os.getenv("CLAUDE_API_KEY")
    
    if not api_key:
        return "Error: CLAUDE_API_KEY not found. Please enter it in the Sponsor Keys panel or set it as an environment variable."
    
    if not ANIMALS_DATA:
        return "Error: No animal data loaded. Please check the server logs."

    # Retrieval Step
    relevant_animals = search_animals(user_input)
    
    if not relevant_animals:
        context_str = "No specific animals found matching the query in the dataset."
    else:
        # Minify JSON to save tokens
        context_str = json.dumps(relevant_animals, ensure_ascii=False)

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

    # Async HTTP Request
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                CLAUDE_API_URL, 
                headers=headers, 
                json=payload, 
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("content", [])
                if content and len(content) > 0:
                    return content[0].get("text", "No text returned.")
                else:
                    return "Empty response from Claude."
            elif response.status_code == 429:
                return "Error: Rate limit exceeded. Please try again in a moment."
            else:
                return f"API Error {response.status_code}: {response.text}"

        except httpx.TimeoutException:
            return "Error: Request timed out. Claude is taking too long to respond."
        except httpx.RequestError as e:
            return f"Network Error: {str(e)}"
        except Exception as e:
            return f"Unexpected Error: {str(e)}"
