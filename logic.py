import json
import os
import httpx
import asyncio
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import modal_ops  # Import the Modal app definition

# --- Configuration ---
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-haiku-20240307"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """
You are a knowledgeable AI assistant specialized in animals. 
When answering user queries, follow these steps:

1. First, check all available datasets (merged animals.json + any enriched datasets) for relevant information.
2. Use the dataset information to answer factual parts of the query.
3. If the dataset does not fully answer the query, use your internal knowledge to provide a complete answer.
4. Clearly indicate which parts of your answer come from the dataset and which parts are generated from your knowledge.
5. Provide concise, readable, and structured answers.
6. Include relevant traits, diet, habitat, and fun facts from datasets where available.
"""

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

# --- Demo Keys ---
# These allow users to try the app without their own keys, subject to rate limits.
DEMO_KEYS = {
    "samba": os.getenv("DEMO_SAMBANOVA_KEY"),
    "claude": os.getenv("DEMO_CLAUDE_KEY"),
    "modal": os.getenv("DEMO_MODAL_KEY"),
    "blaxel": os.getenv("DEMO_BLAXEL_KEY"),
    "gemini": os.getenv("DEMO_GEMINI_KEY")
}

# --- SambaNova Model Catalog ---
DEFAULT_SAMBANOVA_MODEL = "Meta-Llama-3.3-70B-Instruct"
SAMBANOVA_MODEL_IDS = {
    "Meta-Llama-3.3-70B-Instruct": "Meta-Llama-3.3-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct",
    "DeepSeek-R1-0528": "DeepSeek-R1-0528",
    "DeepSeek-R1-Distill-Llama-70B": "DeepSeek-R1-Distill-Llama-70B",
    "DeepSeek-V3-0324": "DeepSeek-V3-0324",
    "DeepSeek-V3.1": "DeepSeek-V3.1"
}

# Map legacy or human-friendly labels to the exact model IDs above.
SAMBANOVA_ALIAS_MAP = {
    "Llama 3.3 70B": "Meta-Llama-3.3-70B-Instruct",
    "Llama 3.1 8B": "Meta-Llama-3.1-8B-Instruct",
    "DeepSeek R1": "DeepSeek-R1-0528",
    "DeepSeek R1 Distill": "DeepSeek-R1-Distill-Llama-70B",
    "DeepSeek V3": "DeepSeek-V3-0324",
    "DeepSeek V3.1": "DeepSeek-V3.1"
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

def format_thinking_process(text: str) -> str:
    """
    Formats the thinking process in the model's response.
    Supports <think> tags and > blockquote style.
    """
    # Check for <think> tags (Standard DeepSeek R1)
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return f"<details><summary>🧠 Thinking Process</summary>\n\n{thinking}\n</details>\n\n{answer}"
    
    # Check for > blockquote style (User's specific case)
    # Pattern: Starts with > and separated by a standalone > line
    if text.strip().startswith(">"):
        if "\n>\n" in text:
            parts = text.split("\n>\n", 1)
            thinking = parts[0].strip().lstrip(">").strip()
            answer = parts[1].strip()
            return f"<details><summary>🧠 Thinking Process</summary>\n\n{thinking}\n</details>\n\n{answer}"
            
    return text

def validate_keys() -> Dict[str, bool]:
    """
    Returns a dictionary indicating which keys are present.
    """
    return {k: v is not None for k, v in SESSION_KEYS.items()}

def get_samba_key():
    """
    Returns the SambaNova API key if present.
    Returns: (key, is_demo)
    """
    key = SESSION_KEYS.get("samba") or os.getenv("SAMBANOVA_API_KEY")
    if key:
        return key, False
        
    demo_key = DEMO_KEYS.get("samba")
    if demo_key:
        return demo_key, True
        
    raise ValueError("SambaNova API Key is missing.")

def get_claude_key():
    """
    Returns the Claude API key if present.
    Returns: (key, is_demo)
    """
    key = SESSION_KEYS.get("claude") or os.getenv("CLAUDE_API_KEY")
    if key:
        return key, False
        
    demo_key = DEMO_KEYS.get("claude")
    if demo_key:
        return demo_key, True
        
    raise ValueError("Claude API Key is missing.")

def get_blaxel_key():
    """
    Returns the Blaxel API key if present.
    Returns: (key, is_demo)
    """
    key = SESSION_KEYS.get("blaxel") or os.getenv("BLAXEL_API_KEY")
    if key:
        return key, False
        
    demo_key = DEMO_KEYS.get("blaxel")
    if demo_key:
        return demo_key, True
        
    raise ValueError("Blaxel API Key is missing.")

def get_gemini_key():
    """
    Returns the Google Gemini API key if present.
    Returns: (key, is_demo)
    """
    key = SESSION_KEYS.get("gemini") or os.getenv("GEMINI_API_KEY")
    if key:
        return key, False
        
    demo_key = DEMO_KEYS.get("gemini")
    if demo_key:
        return demo_key, True
        
    raise ValueError("Google Gemini API Key is missing.")

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
    # Format: "Name: [name]. Description: [description]. Traits: [traits]"
    texts = []
    for animal in data:
        text = f"Name: {animal.get('name', '')}. Description: {animal.get('description', '')}"
        
        # Add traits if available
        traits = animal.get('traits', [])
        if isinstance(traits, list) and traits:
            text += f". Traits: {', '.join(str(t) for t in traits)}"
        elif isinstance(traits, dict) and traits:
            text += f". Traits: {', '.join(f'{k}: {v}' for k, v in traits.items())}"
            
        # Add classification if available as top-level fields
        classification_fields = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        class_parts = []
        for field in classification_fields:
            if animal.get(field):
                class_parts.append(f"{field.capitalize()}: {animal.get(field)}")
        if class_parts:
            text += f". Classification: {', '.join(class_parts)}"

        # Add physiological traits if available
        if animal.get('body_mass'):
            text += f". Body Mass: {animal.get('body_mass')}"
        if animal.get('metabolic_rate'):
            text += f". Metabolic Rate: {animal.get('metabolic_rate')}"
        if animal.get('brain_size'):
            text += f". Brain Size: {animal.get('brain_size')}"
            
        texts.append(text)
    
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
    elif DEMO_KEYS.get("modal"):
        # Try demo key
        try:
            token_id, token_secret = DEMO_KEYS["modal"].split(":", 1)
            os.environ["MODAL_TOKEN_ID"] = token_id.strip()
            os.environ["MODAL_TOKEN_SECRET"] = token_secret.strip()
            use_modal = True
            print("Using Demo Modal Token.")
        except Exception as e:
            print(f"Invalid Demo Modal token format: {e}")

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
                
                # Filter out results with "null values" (incomplete data)
                # If description is missing or generic, and no other key info exists, skip it.
                desc = item.get('description', '')
                is_placeholder = desc == "Imported from external database."
                has_diet = bool(item.get('diet'))
                has_fun_fact = bool(item.get('fun_fact'))
                # Check for traits (list or dict)
                traits = item.get('traits')
                has_traits = bool(traits) if isinstance(traits, (list, dict)) else False
                
                # Also check physiological data
                has_physio = bool(item.get('body_mass') or item.get('metabolic_rate') or item.get('brain_size'))

                if (not desc or is_placeholder) and not (has_diet or has_fun_fact or has_traits or has_physio):
                    continue

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

async def run_samba(prompt: str, model_choice: Optional[str]) -> str:
    """Call SambaNova's OpenAI-compatible endpoint using httpx."""
    # Resolve model selection with sane defaults.
    selected_key = (model_choice or "").strip() or DEFAULT_SAMBANOVA_MODEL
    model_id = SAMBANOVA_MODEL_IDS.get(selected_key) or SAMBANOVA_ALIAS_MAP.get(selected_key)
    if not model_id:
        raise ValueError(f"Invalid SambaNova model selection: {selected_key}")

    # Fetch the API key (raises ValueError if missing to align with UX).
    api_key, is_demo = get_samba_key()

    url = "https://api.sambanova.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Use a concise system prompt so responses stay focused on the animal data.
    system_prompt = (
        "You are an expert biologist and data analyst. "
        "Provide a detailed answer based strictly on the supplied context. "
        "Offer structured reasoning when the user asks for it."
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "top_p": 0.1
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if is_demo:
            raise RuntimeError(f"Demo key for SambaNova failed. Please provide your own API key to use this model reliably. (Error: {e.response.status_code})")
        error_text = e.response.text if e.response else str(e)
        raise RuntimeError(f"SambaNova API Error ({e.response.status_code}): {error_text}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"SambaNova network error: {str(e)}") from e

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("SambaNova API returned no choices.")

    message = choices[0].get("message", {})
    text = message.get("content") or choices[0].get("text", "")
    if not text:
        raise RuntimeError("SambaNova API returned empty content.")

    # Format thinking process if present
    formatted_text = format_thinking_process(text.strip())

    return f"Powered by SambaNova ({model_id})\n\n{formatted_text}"

async def run_claude(prompt: str, model_version: str) -> str:
    """
    Executes the prompt using Anthropic's Claude API via httpx.
    Raises exceptions on failure so the caller can handle fallbacks.
    """
    # 1. Get Key (will raise ValueError if missing)
    api_key, is_demo = get_claude_key()
    
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
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            # Check for 400 error specifically to handle credit balance issues gracefully
            if response.status_code == 400 and "credit balance is too low" in response.text:
                 raise ValueError("Anthropic API Credit Balance Too Low. Please top up your account.")
                 
            response.raise_for_status()
            text = response.json()["content"][0]["text"]
    except httpx.HTTPStatusError as e:
        if is_demo:
            raise RuntimeError(f"Demo key for Claude failed. Please provide your own API key to use this model reliably. (Error: {e.response.status_code})")
        raise e
        
    # 5. Format Output
    return text

# --- Google Model Mapping ---
# Maps UI labels to the exact API IDs provided by the user.
GOOGLE_MODEL_MAP = {
    "Gemini 3.0 Pro Preview": "gemini-3-pro-preview",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
    "Gemini 2.5 Flash Image": "gemini-2.5-flash-image",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 2.0 Flash-Lite": "gemini-2.0-flash-lite"
}

async def run_gemini(prompt: str, model_version: str) -> str:
    """
    Executes the prompt using Google's Gemini API via httpx.
    Supports the new Gemini 2.5/3.0 models.
    """
    try:
        api_key, is_demo = get_gemini_key()
        
        # Resolve Model ID from the map
        model_id = GOOGLE_MODEL_MAP.get(model_version)
        
        # Fallback if mapping fails (should not happen if UI is synced)
        if not model_id:
            model_id = "gemini-2.5-flash" # Default to stable flash
            
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
                if is_demo:
                     # If demo key is exhausted, suggest using own key
                     return (
                         f"### ⚠️ Demo Key Limit Reached\n"
                         f"The shared demo key for **{model_version}** is currently rate-limited. "
                         f"Please try again later or enter your own Google API key in the sidebar for stable testing."
                     )

                print(f"Gemini 429 Error for {model_id}: {response.text}")
                
                # Fallback Strategy: Try gemini-2.5-flash-lite (Cheapest/Fastest)
                if model_id != "gemini-2.5-flash-lite":
                    print("Falling back to Gemini 2.5 Flash-Lite...")
                    fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
                    response = await client.post(fallback_url, headers=headers, json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response text found.")
                        return f"### 💎 Powered by Google Gemini (Fallback to Flash-Lite)\n> *(Original model {model_version} quota exceeded)*\n\n{text}"
            
            if response.status_code != 200:
                if is_demo:
                     return (
                         f"### ⚠️ Demo Key Error\n"
                         f"The demo key failed for **{model_version}** (Error {response.status_code}). "
                         f"Please provide your own Google API key."
                     )
                # Return the raw error for debugging if not demo
                return f"Gemini API Error ({response.status_code}): {response.text}"

            response.raise_for_status()
            result = response.json()
            # Extract text from Gemini response structure
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response text found.")
            
        return text
            
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
        api_key, is_demo = get_blaxel_key()
        
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
            
            if response.status_code != 200 and is_demo:
                 raise RuntimeError(f"Demo key for Blaxel failed. Please provide your own API key. (Error: {response.status_code})")

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

async def get_random_animal_fact(provider: str) -> str:
    """
    Generates a random animal fact using the selected provider.
    """
    if not ANIMALS_DATA:
        return "⚠️ No animal data loaded to pick from."
    
    try:
        # Pick a random animal
        animal = random.choice(ANIMALS_DATA)
        name = animal.get("name", "Unknown Animal")
        
        prompt = (
            f"Tell me a fascinating, short 'Did You Know?' fact about the {name}. "
            "Keep it under 50 words and make it engaging."
        )
        
        response_text = ""
        badge = ""

        # Route to provider (simplified)
        if "SambaNova" in provider:
            # Extract version from provider string
            version = provider.split("–")[-1].strip() if "–" in provider else provider
            model_choice = SAMBANOVA_ALIAS_MAP.get(version) or version
            return await run_samba(prompt, model_choice)
        elif "Gemini" in provider:
            badge = "### 💎 Powered by Google Gemini"
            response_text = await run_gemini(prompt, "1.5 Flash")
        elif "Claude" in provider:
            badge = "### 🧠 Powered by Anthropic Claude"
            response_text = await run_claude(prompt, "Haiku")
        elif "Blaxel" in provider:
            badge = "### 🚀 Powered by Blaxel"
            response_text = await run_blaxel(prompt)
        else:
            # Fallback to local description for offline use.
            badge = "### 📂 Local Data"
            desc = animal.get('description', 'No description available.')
            response_text = f"**Random Fact about {name}:** {desc[:200]}..."

        return f"{badge}\n\n{response_text}"
             
    except Exception as e:
        return f"❌ Failed to generate fact: {str(e)}"

async def run_model(
    provider: str,
    user_input: str,
    use_blaxel: bool = False
) -> str:
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
        
        # Format as readable text with key fields
        formatted_context = []
        for idx, animal in enumerate(relevant_animals):
            info = f"{idx+1}. **{animal.get('name', 'Unknown')}**\n"
            if animal.get('description'):
                info += f"   - Description: {animal.get('description')}\n"
            if animal.get('diet'):
                info += f"   - Diet: {animal.get('diet')}\n"
            if animal.get('habitat'):
                info += f"   - Habitat: {animal.get('habitat')}\n"
            if animal.get('traits'):
                info += f"   - Traits: {animal.get('traits')}\n"
            if animal.get('fun_fact'):
                info += f"   - Fun Fact: {animal.get('fun_fact')}\n"
            
            # Add physiological data if present
            physio = []
            if animal.get('body_mass'): physio.append(f"Mass: {animal.get('body_mass')}g")
            if animal.get('metabolic_rate'): physio.append(f"Metabolic Rate: {animal.get('metabolic_rate')}W")
            if physio:
                info += f"   - Physiology: {', '.join(physio)}\n"
                
            formatted_context.append(info)
        
        context_str = f"**Search Source:** {source_label}\n\n" + "\n".join(formatted_context)

    # 2. Construct Prompt with Context
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"[Dataset Context]:\n{context_str}\n\n"
        f"[User Query]:\n{user_input}\n"
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
    badge: Optional[str] = ""
    print(f"DEBUG: Routing to provider: {provider}")
    
    try:
        if provider.startswith("SambaNova"):
            badge = None
            # Extract version for backwards compatibility, but prefer explicit selection.
            version = provider.split("–")[-1].strip() if "–" in provider else provider
            model_choice = SAMBANOVA_ALIAS_MAP.get(version) or version
            try:
                main_response = await run_samba(full_prompt, model_choice)
            except Exception as e:
                # Fallback for SambaNova
                print(f"SambaNova API failed: {e}")
                resolved_label = (
                    SAMBANOVA_MODEL_IDS.get(model_choice)
                    or SAMBANOVA_ALIAS_MAP.get(model_choice)
                    or DEFAULT_SAMBANOVA_MODEL
                )
                badge = f"### ⚡ Powered by SambaNova ({resolved_label})"
                main_response = (
                    f"SambaNova is unavailable: {str(e)}\n\n"
                    f"Showing keyword search result instead:\n\n{context_str}"
                )
        
        elif provider.startswith("Claude"):
            badge = "### 🧠 Powered by Anthropic Claude"
            version = provider.split("–")[-1].strip()
            try:
                main_response = await run_claude(full_prompt, version)
            except Exception as e:
                # Fallback Logic: Try SambaNova if Claude fails
                print(f"Claude API failed: {e}. Attempting fallback to SambaNova...")
                try:
                    # Fallback to a capable SambaNova model (Llama 3.3 70B)
                    fallback_result = await run_samba(full_prompt, DEFAULT_SAMBANOVA_MODEL)
                    badge = "### ⚡ Powered by SambaNova Cloud (Fallback from Claude)"
                    main_response = (
                        f"> *(Claude unavailable: {str(e)})*\n\n"
                        f"{fallback_result.replace('Powered by SambaNova', '').strip()}"
                    )
                except Exception as samba_e:
                    # If fallback also fails, show keyword search
                    badge = "### ❌ Service Unavailable"
                    main_response = (
                        f"Claude Error: {str(e)}\n"
                        f"SambaNova Error: {str(samba_e)}\n\n"
                        f"Showing keyword search result instead:\n\n{context_str}"
                    )
        
        elif provider.startswith("Google"):
            badge = "### 💎 Powered by Google"
            print("DEBUG: Calling Google Model...")
            # Robust split for version
            if "–" in provider:
                version = provider.split("–")[-1].strip()
            elif "-" in provider:
                version = provider.split("-")[-1].strip()
            else:
                version = provider
            main_response = await run_gemini(full_prompt, version)
        
        elif provider.startswith("Blaxel"):
            badge = "### 🚀 Powered by Blaxel"
            print("DEBUG: Calling Blaxel (Main)...")
            main_response = await run_blaxel(full_prompt)
        
        elif provider.startswith("Local"):
            badge = "### 📂 Local Model"
            main_response = run_local(full_prompt)
        
        else:
            badge = "### ❓ Unknown Provider"
            main_response = f"Error: Unknown provider '{provider}'"
            
    except Exception as e:
        badge = "### ⚠️ Error"
        main_response = f"Routing Error: {str(e)}"

    # 5. Combine Results
    output_parts = []
    if badge:
        output_parts.append(badge)
    if main_response:
        output_parts.append(main_response)
    final_output = "\n\n".join(output_parts)

    if blaxel_task:
        try:
            blaxel_result = await blaxel_task
            # Check if blaxel_result is an error message
            if "Error" not in blaxel_result and "Configuration Error" not in blaxel_result:
                final_output += f"\n\n---\n\n### ✨ Suggested by Blaxel (Sponsor)\n\n{blaxel_result}"
            else:
                print(f"Blaxel task returned error: {blaxel_result}")
        except Exception as e:
            print(f"Blaxel task failed: {e}")

    return final_output

# Deprecated: Kept for backward compatibility if needed, but run_model should be used.
async def query_claude(user_input: str) -> str:
    return await run_model("Claude – Haiku", user_input, False)

