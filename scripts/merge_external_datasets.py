import json
import pandas as pd
import os
import sys

# Add parent directory to path to import logic if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def merge_datasets():
    print("🔄 Starting data merge process...")
    
    # Paths
    base_path = "data/animals.json"
    ext_class_path = "data/external/classification.csv"
    ext_traits_path = "data/external/traits.csv"
    output_path = "data/animals_enriched.json"

    # 1. Load Base Dataset
    if not os.path.exists(base_path):
        print(f"❌ Base file {base_path} not found.")
        return

    with open(base_path, 'r') as f:
        base_data = json.load(f)
    
    df_base = pd.DataFrame(base_data)
    print(f"✅ Loaded {len(df_base)} records from {base_path}")

    # 2. Load External Datasets
    # Classification Data
    if os.path.exists(ext_class_path):
        df_class = pd.read_csv(ext_class_path)
        df_class.rename(columns={'scientific_name': 'name'}, inplace=True)
        print(f"✅ Loaded {len(df_class)} classification records")
    else:
        print(f"⚠️ Classification file {ext_class_path} not found. Skipping.")
        df_class = pd.DataFrame()

    # Traits Data
    if os.path.exists(ext_traits_path):
        df_traits = pd.read_csv(ext_traits_path)
        # Map external fields to internal schema
        df_traits.rename(columns={
            'scientific_name': 'name',
            'diet_type': 'diet',
            'average_lifespan': 'lifespan',
            'habitat_type': 'habitat'
        }, inplace=True)
        print(f"✅ Loaded {len(df_traits)} trait records")
    else:
        print(f"⚠️ Traits file {ext_traits_path} not found. Skipping.")
        df_traits = pd.DataFrame()

    # 3. Merge Datasets
    # Merge Classification
    if not df_class.empty:
        # Left join to keep all base animals, add classification info where matches found
        df_base = pd.merge(df_base, df_class, on='name', how='left')
        print("🔹 Merged classification data")

    # Merge Traits
    if not df_traits.empty:
        # We want to fill missing values in base with values from traits
        # First merge
        df_base = pd.merge(df_base, df_traits, on='name', how='left', suffixes=('', '_ext'))
        
        # Fill missing fields
        for field in ['diet', 'lifespan', 'habitat']:
            ext_col = f'{field}_ext'
            if ext_col in df_base.columns:
                if field in df_base.columns:
                    # Fill NaN in base with value from ext
                    df_base[field] = df_base[field].fillna(df_base[ext_col])
                    # Also fill empty strings if any
                    df_base[field] = df_base[field].replace('', pd.NA).fillna(df_base[ext_col])
                else:
                    # If field didn't exist in base, create it
                    df_base[field] = df_base[ext_col]
                
                # Drop the temporary external column
                df_base.drop(columns=[ext_col], inplace=True)
        
        # Handle behavior_traits
        if 'behavior_traits' in df_base.columns:
            # If 'traits' column exists and is a list, we might want to append
            # But for now, let's just keep behavior_traits as a separate field or merge text
            pass
            
        print("🔹 Merged traits data")

    # 4. Deduplicate
    initial_count = len(df_base)
    df_base.drop_duplicates(subset=['name'], inplace=True)
    if len(df_base) < initial_count:
        print(f"🔹 Removed {initial_count - len(df_base)} duplicate entries")

    # 5. Convert back to JSON structure
    final_data = df_base.to_dict(orient='records')
    
    # Clean up: Remove NaN/None values to keep JSON clean
    cleaned_data = []
    for record in final_data:
        clean_record = {}
        for k, v in record.items():
            # Handle lists (like traits, source_urls)
            if isinstance(v, list):
                clean_record[k] = v
            # Handle scalars
            elif pd.notna(v) and v != "":
                clean_record[k] = v
        cleaned_data.append(clean_record)

    # 6. Fallthrough: Check for missing fields and simulate enrichment call
    # In a real HF Space environment without Modal, we might skip this or use a local fallback.
    # Here we just log what's missing.
    missing_info_count = 0
    for animal in cleaned_data:
        if not animal.get('fun_fact') or not animal.get('diet'):
            missing_info_count += 1
    
    if missing_info_count > 0:
        print(f"⚠️ {missing_info_count} animals are still missing key fields (fun_fact, diet).")
        print("💡 Tip: Run 'modal run scripts/enrich_animals_modal.py' to fill these gaps using LLMs.")

    # 7. Save Final File
    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"✅ Successfully saved enriched dataset to {output_path}")
    print(f"📊 Final Record Count: {len(cleaned_data)}")

if __name__ == "__main__":
    merge_datasets()
