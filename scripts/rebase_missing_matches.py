
import json
import os
import pandas as pd

def rebase_missing_matches():
    enriched_path = "data/animals_enriched.json"
    missing_log_path = "data/external/missing_matches.log"

    if not os.path.exists(enriched_path):
        print(f"Error: {enriched_path} not found.")
        return

    print(f"Loading {enriched_path}...")
    with open(enriched_path, 'r') as f:
        data = json.load(f)
    
    print(f"Analyzing {len(data)} records...")
    
    missing_matches = []
    
    # Classification columns to check
    class_cols = ['phylum', 'class', 'order', 'family', 'genus']
    # Trait columns to check
    trait_cols = ['diet', 'habitat', 'lifespan', 'behavior']

    for animal in data:
        # Check if any classification info exists
        has_class = any([col in animal and animal[col] for col in class_cols])
        
        # Check if any trait info exists
        has_trait = any([col in animal and animal[col] for col in trait_cols])
        
        # If neither exists, it's a "missing match" (i.e., not enriched)
        if not has_class and not has_trait:
            missing_matches.append(animal.get('name', 'Unknown'))

    print(f"Found {len(missing_matches)} animals with missing enrichment data.")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(missing_log_path), exist_ok=True)
    
    with open(missing_log_path, 'w') as f:
        f.write('\n'.join(missing_matches))
        
    print(f"Updated {missing_log_path}")

if __name__ == "__main__":
    rebase_missing_matches()
