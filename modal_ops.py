import modal
import numpy as np

# Define the Modal App
app = modal.App("astrx-mcp-embeddings")

# Define the image with dependencies
image = modal.Image.debian_slim().pip_install("sentence-transformers", "numpy")

@app.function(image=image)
def generate_embeddings(texts: list[str]):
    from sentence_transformers import SentenceTransformer
    
    # Load model inside the container (cached if possible)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Compute embeddings
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings
