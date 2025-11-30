import json
import os

def merge_datasets():
    original_path = "data/animals.json"
    enriched_path = "data/animals_enriched.json"
    
    if not os.path.exists(original_path) or not os.path.exists(enriched_path):
        print("One or both data files not found.")
        return

    print(f"Loading original data from {original_path}...")
    with open(original_path, "r") as f:
        original_data = json.load(f)
    
    print(f"Loading enriched data from {enriched_path}...")
    with open(enriched_path, "r") as f:
        enriched_data = json.load(f)
        
    print(f"Original count: {len(original_data)}")
    print(f"Enriched count: {len(enriched_data)}")
    
    # Create a map for faster lookup
    # Assuming 'name' is the unique identifier
    enriched_map = {item.get('name'): item for item in enriched_data if item.get('name')}
    
    merged_count = 0
    final_data = []
    
    for item in original_data:
        name = item.get('name')
        if name in enriched_map:
            # Use the enriched version
            final_data.append(enriched_map[name])
            merged_count += 1
        else:
            # Keep the original
            final_data.append(item)
            
    print(f"Merged {merged_count} enriched records into the main dataset.")
    print(f"Final dataset size: {len(final_data)}")
    
    # Save back to animals.json
    print(f"Saving merged data to {original_path}...")
    with open(original_path, "w") as f:
        json.dump(final_data, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    merge_datasets()
