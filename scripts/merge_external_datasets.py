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
    ext_class_path = "data/animals-external.csv"
    ext_traits_path = "data/Animal Traits Observations.csv"
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
        try:
            df_class = pd.read_csv(ext_class_path, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"⚠️ UTF-8 failed for {ext_class_path}, trying latin1.")
            df_class = pd.read_csv(ext_class_path, encoding='latin1')
            
        # Map columns: Scientific.Name -> name, and lower case columns
        df_class.rename(columns={
            'Scientific.Name': 'name',
            'Kingdom': 'kingdom',
            'Phylum': 'phylum',
            'Class': 'class',
            'Order': 'order',
            'Family': 'family',
            'Genus': 'genus'
        }, inplace=True)
        print(f"✅ Loaded {len(df_class)} classification records")
    else:
        print(f"⚠️ Classification file {ext_class_path} not found. Skipping.")
        df_class = pd.DataFrame()

    # Traits Data
    if os.path.exists(ext_traits_path):
        try:
            df_traits = pd.read_csv(ext_traits_path, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"⚠️ UTF-8 failed for {ext_traits_path}, trying latin1.")
            df_traits = pd.read_csv(ext_traits_path, encoding='latin1')
            
        # Map external fields to internal schema
        # The file has 'species' which seems to be the scientific name
        df_traits.rename(columns={
            'species': 'name',
            'body mass': 'body_mass',
            'metabolic rate': 'metabolic_rate',
            'brain size': 'brain_size'
        }, inplace=True)
        print(f"✅ Loaded {len(df_traits)} trait records")
    else:
        print(f"⚠️ Traits file {ext_traits_path} not found. Skipping.")
        df_traits = pd.DataFrame()

    # 3. Merge Datasets
    # Merge Classification
    if not df_class.empty:
        # Outer join to include animals from classification file that aren't in base
        df_base = pd.merge(df_base, df_class, on='name', how='outer')
        print("🔹 Merged classification data")

    # Merge Traits
    if not df_traits.empty:
        # Outer join to include animals from traits file
        df_base = pd.merge(df_base, df_traits, on='name', how='outer', suffixes=('', '_ext'))
        
        # Fields to merge from traits
        trait_fields = ['body_mass', 'metabolic_rate', 'brain_size']
        
        for field in trait_fields:
            # If the field exists in the merged dataframe (from traits), ensure it's in the final
            if field in df_base.columns:
                # If it was already in base (unlikely for these specific ones, but good practice)
                pass 
            elif f'{field}_ext' in df_base.columns:
                 df_base[field] = df_base[f'{field}_ext']
                 df_base.drop(columns=[f'{field}_ext'], inplace=True)
            # If it's just in traits (which merged into base), it might already be there as 'field' 
            # if base didn't have it.
            
        # Clean up any _ext columns that might have been created if base had these columns
        for col in df_base.columns:
            if col.endswith('_ext'):
                base_col = col[:-4]
                if base_col in df_base.columns:
                    df_base[base_col] = df_base[base_col].fillna(df_base[col])
                else:
                    df_base[base_col] = df_base[col]
                df_base.drop(columns=[col], inplace=True)
            
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
