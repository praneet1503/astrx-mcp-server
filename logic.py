import json
import os
import httpx
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import modal_ops  # Import the Modal app definition

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
    "blaxel": None,
    "gemini": None
}

def save_keys(samba_key: str, claude_key: str, modal_key: str, blaxel_key: str, gemini_key: str) -> str:
    """
    Saves the provided API keys to the session storage.
    """
    global SESSION_KEYS
    SESSION_KEYS["samba"] = samba_key if samba_key and samba_key.strip() else None
    SESSION_KEYS["claude"] = claude_key if claude_key and claude_key.strip() else None
    SESSION_KEYS["modal"] = modal_key if modal_key and modal_key.strip() else None
    SESSION_KEYS["blaxel"] = blaxel_key if blaxel_key and blaxel_key.strip() else None
    SESSION_KEYS["gemini"] = gemini_key if gemini_key and gemini_key.strip() else None
    
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
    key = SESSION_KEYS.get("blaxel") or os.getenv("BLAXEL_API_KEY")
    if not key:
        raise ValueError("Blaxel API Key is missing.")
    return key

def get_gemini_key():
    """
    Returns the Google Gemini API key if present.
    """
    key = SESSION_KEYS.get("gemini") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Google Gemini API Key is missing.")
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

async def search_animals(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Semantic search using vector embeddings.
    Uses Modal if configured, otherwise falls back to local SentenceTransformer.
    """
    if not query:
        return []

    # Check for Modal Token
    modal_token = SESSION_KEYS.get("modal")
    use_modal = False
    
    if modal_token and ":" in modal_token:
        try:
            token_id, token_secret = modal_token.split(":", 1)
            os.environ["MODAL_TOKEN_ID"] = token_id.strip()
            os.environ["MODAL_TOKEN_SECRET"] = token_secret.strip()
            use_modal = True
        except Exception as e:
            print(f"Invalid Modal token format: {e}")

    # Fallback to keyword search if semantic search is not ready AND Modal is not used
    if not use_modal and (RETRIEVER_MODEL is None or ANIMAL_EMBEDDINGS is None):
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
        query_embedding = None
        
        if use_modal:
            print("Using Modal for embedding generation...")
            try:
                # Run the Modal function ephemerally
                with modal_ops.app.run():
                    query_embedding = modal_ops.generate_embeddings.remote([query])[0]
                print("Modal embedding generated successfully.")
            except Exception as e:
                print(f"Modal execution failed: {e}")
                # Fallback to local if Modal fails
                if RETRIEVER_MODEL:
                    query_embedding = RETRIEVER_MODEL.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
        # If Modal wasn't used or failed, use local model
        if query_embedding is None and RETRIEVER_MODEL:
             query_embedding = RETRIEVER_MODEL.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        if query_embedding is None:
             return [] # Should have fallen back to keyword search earlier if local model was missing

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
                # Mark if Modal was used (for UI display)
                if use_modal:
                    item['_source'] = "Modal Semantic Search"
                else:
                    item['_source'] = "Local Semantic Search"
                results.append(item)
                
        return results
        
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []

# --- Model Runners (Stubs) ---

async def run_samba(prompt: str, model_version: str) -> str:
    """
    Executes the prompt using SambaNova's API via OpenAI client.
    Raises exceptions on failure so the caller can handle fallbacks.
    """
    # 1. Get Key (will raise ValueError if missing)
    api_key = get_samba_key()
    
    # 2. Map Model ID
    model = "Meta-Llama-3.1-8B-Instruct"
    if "405B" in model_version:
        model = "Meta-Llama-3.1-405B-Instruct"
    elif "3.3 70B" in model_version:
        model = "Meta-Llama-3.3-70B-Instruct"
    elif "3.1 70B" in model_version:
        model = "Meta-Llama-3.1-70B-Instruct"
    elif "DeepSeek R1" in model_version:
        model = "DeepSeek-R1"
    elif "DeepSeek R1 Distill" in model_version:
        model = "DeepSeek-R1-Distill-Llama-70B"
        
    # 3. Initialize OpenAI Client with SambaNova Base URL
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.sambanova.ai/v1",
    )
    
    # Enhanced system prompt for "advanced tasks"
    system_prompt = (
        "You are an expert biologist and data analyst. "
        "Provide a detailed answer based on the context. "
        "If the user asks for reasoning or summary, provide a structured response."
    )
    
    # 4. Execute Async Call
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        top_p=0.1
    )
    
    text = response.choices[0].message.content
        
    # 5. Format Output
    return f"### ⚡ Powered by SambaNova Cloud\n\n{text}"

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
        # Check for 400 error specifically to handle credit balance issues gracefully
        if response.status_code == 400 and "credit balance is too low" in response.text:
             raise ValueError("Anthropic API Credit Balance Too Low. Please top up your account.")
             
        response.raise_for_status()
        text = response.json()["content"][0]["text"]
        
    # 5. Format Output
    return f"### 🧠 Powered by Anthropic Claude\n\n{text}"

async def run_gemini(prompt: str, model_version: str = "1.5 Flash") -> str:
    """
    Executes the prompt using Google's Gemini API via httpx.
    """
    try:
        api_key = get_gemini_key()
        
        model_id = "gemini-1.5-flash-latest"
        if "3.0" in model_version:
            model_id = "gemini-3-pro-preview"
        elif "2.5 Pro" in model_version:
            model_id = "gemini-2.5-pro"
        elif "2.5 Flash-Lite" in model_version:
            model_id = "gemini-2.5-flash-lite"
        elif "2.5 Flash" in model_version:
            model_id = "gemini-2.5-flash"
        elif "1.5 Pro" in model_version:
            model_id = "gemini-1.5-pro-latest"
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            # Handle 429 Resource Exhausted (Quota Limit)
            if response.status_code == 429:
                print(f"Gemini 429 Error for {model_id}: {response.text}")
                # If it's a high-end model, try falling back to Flash
                if model_id != "gemini-1.5-flash-latest":
                    print("Falling back to Gemini 1.5 Flash...")
                    fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
                    response = await client.post(fallback_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response text found.")
                        return f"### 💎 Powered by Google Gemini (Fallback to 1.5 Flash)\n> *(Original model {model_version} quota exceeded)*\n\n{text}"
            
            response.raise_for_status()
            result = response.json()
            # Extract text from Gemini response structure
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response text found.")
            
        return f"### 💎 Powered by Google Gemini\n\n{text}"
            
    except ValueError as e:
        return f"Configuration Error: {str(e)}"
    except httpx.HTTPStatusError as e:
        return f"Gemini API Error: {e.response.text}"
    except Exception as e:
        return f"Gemini Error: {str(e)}"

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

async def run_model(provider: str, user_input: str, use_blaxel: bool = False) -> str:
    """
    Unified routing function to dispatch requests to the selected provider.
    """
    if not user_input or not user_input.strip():
        return "Please enter a valid question."

    # 1. Retrieve Context (RAG)
    if not ANIMALS_DATA:
        return "Error: No animal data loaded."

    relevant_animals = await search_animals(user_input)
    if not relevant_animals:
        context_str = "No specific animals found matching the query."
    else:
        # Add source info to context
        source_label = relevant_animals[0].get('_source', 'Unknown Source')
        context_str = f"[Search Source: {source_label}]\n" + json.dumps(relevant_animals, ensure_ascii=False)

    # 2. Construct Prompt with Context
    full_prompt = (
        f"Context:\n{context_str}\n\n"
        f"User Question: {user_input}\n\n"
        "Answer based on the context provided."
    )

    # 3. Prepare Blaxel Task (if enabled)
    blaxel_task = None
    print(f"DEBUG: use_blaxel={use_blaxel}")
    if use_blaxel:
        try:
            # Check if key exists
            get_blaxel_key()
            print("DEBUG: Blaxel key found, starting task.")
            # Create a specific prompt for suggestions
            blaxel_prompt = (
                f"User Query: {user_input}\n"
                f"Context Summary: {context_str[:1000]}\n\n"
                "Provide 3 short, fascinating 'Did You Know?' facts or insights related to this animal/topic. "
                "Format as a bulleted list."
            )
            blaxel_task = asyncio.create_task(run_blaxel(blaxel_prompt))
        except ValueError:
            # Key missing, ignore Blaxel
            print("DEBUG: Blaxel key missing, skipping.")
            pass
        except Exception as e:
            print(f"Failed to start Blaxel task: {e}")

    # 4. Route to Main Provider
    main_response = ""
    print(f"DEBUG: Routing to provider: {provider}")
    try:
        if provider.startswith("SambaNova"):
            # Extract version if needed, e.g., "SambaNova – Samba-1" -> "Samba-1"
            version = provider.split("–")[-1].strip()
            try:
                main_response = await run_samba(full_prompt, version)
            except Exception as e:
                # Fallback for SambaNova
                print(f"SambaNova API failed: {e}")
                main_response = (
                    f"**Powered by SambaNova Cloud** (Unavailable)\n\n"
                    f"SambaNova is unavailable: {str(e)}\n\n"
                    f"Showing keyword search result instead:\n\n{context_str}"
                )
        
        elif provider.startswith("Claude"):
            version = provider.split("–")[-1].strip()
            try:
                main_response = await run_claude(full_prompt, version)
            except Exception as e:
                # Fallback Logic: Try SambaNova if Claude fails
                print(f"Claude API failed: {e}. Attempting fallback to SambaNova...")
                try:
                    # Fallback to a capable SambaNova model (Llama 3.3 70B)
                    fallback_result = await run_samba(full_prompt, "Llama 3.3 70B")
                    main_response = (
                        f"### ⚡ Powered by SambaNova Cloud (Fallback from Claude)\n"
                        f"> *(Claude unavailable: {str(e)})*\n\n"
                        f"{fallback_result.replace('### ⚡ Powered by SambaNova Cloud', '').strip()}"
                    )
                except Exception as samba_e:
                    # If fallback also fails, show keyword search
                    main_response = (
                        f"### ❌ Powered by Claude (Unavailable)\n"
                        f"### ❌ Fallback to SambaNova (Unavailable)\n\n"
                        f"Claude Error: {str(e)}\n"
                        f"SambaNova Error: {str(samba_e)}\n\n"
                        f"Showing keyword search result instead:\n\n{context_str}"
                    )
        
        elif provider.startswith("Google Gemini"):
            print("DEBUG: Calling Gemini...")
            version = provider.split("–")[-1].strip()
            main_response = await run_gemini(full_prompt, version)
        
        elif provider.startswith("Blaxel"):
            print("DEBUG: Calling Blaxel (Main)...")
            main_response = await run_blaxel(full_prompt)
        
        elif provider.startswith("Local"):
            main_response = run_local(full_prompt)
        
        else:
            main_response = f"Error: Unknown provider '{provider}'"
            
    except Exception as e:
        main_response = f"Routing Error: {str(e)}"

    # 5. Combine Results
    if blaxel_task:
        try:
            blaxel_result = await blaxel_task
            # Check if blaxel_result is an error message
            if "Error" not in blaxel_result and "Configuration Error" not in blaxel_result:
                main_response += f"\n\n---\n\n### ✨ Suggested by Blaxel (Sponsor)\n\n{blaxel_result}"
            else:
                print(f"Blaxel task returned error: {blaxel_result}")
        except Exception as e:
            print(f"Blaxel task failed: {e}")

    return main_response

# Deprecated: Kept for backward compatibility if needed, but run_model should be used.
async def query_claude(user_input: str) -> str:
    return await run_model("Claude – Haiku", user_input, False)

