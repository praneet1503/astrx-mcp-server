import json
import os
from typing import List, Dict, Any

def list_animals() -> List[Dict[str, Any]]:
    """
    Returns a list of animals with their details.
    Each animal includes name, species, description, traits, and source.
    """
    # Path to the data file
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "animals.json")
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback if data file is missing
        return [
            {
                "name": "Error",
                "species": "System",
                "description": "Could not load animal data.",
                "traits": []
            }
        ]
