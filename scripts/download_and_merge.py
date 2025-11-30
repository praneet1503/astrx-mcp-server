import os
import json
import requests
import pandas as pd
import sys

# Configuration
DATA_DIR = "data"
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
ANIMALS_JSON = os.path.join(DATA_DIR, "animals.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "animals_enriched.json")

CLASSIFICATION_URL = "https://www.opendatabay.com/data/ai-ml/db0bb3d0-3019-4764-a9a2-542becc61ea2/download"
TRAITS_URL = "https://www.opendatabay.com/data/ai-ml/8862d6b1-a8ec-45d6-bdb3-4e43f7217a5c/download"

CLASSIFICATION_CSV = os.path.join(EXTERNAL_DIR, "animals-external.csv")
CLASSIFICATION_XLSX = os.path.join(EXTERNAL_DIR, "animals-external.xlsx")
TRAITS_CSV = os.path.join(EXTERNAL_DIR, "Animal Traits Observations.csv")
TRAITS_XLSX = os.path.join(EXTERNAL_DIR, "Animal Traits Observations.xlsx")

def ensure_directories():
    if not os.path.exists(EXTERNAL_DIR):
        os.makedirs(EXTERNAL_DIR)
        print(f"Created directory: {EXTERNAL_DIR}")

def download_file(url, filepath):
    print(f"Downloading {url} to {filepath}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Create dummy file if download fails to allow script to proceed for testing
        if not os.path.exists(filepath):
            print("Creating dummy file for testing...")
            with open(filepath, 'w') as f:
                if "classification" in filepath:
                    f.write("scientific_name,phylum,class,order,family,genus\n")
                else:
                    f.write("scientific_name,diet_type,activity_pattern,habitat_type,behavior_traits,average_lifespan\n")
                if filepath.endswith('.xlsx'):
                    # If the intended file was xlsx but we couldn't download it, create a minimal csv fallback instead
                    csv_fallback = filepath.replace('.xlsx', '.csv')
                    if not os.path.exists(csv_fallback):
                        with open(csv_fallback, 'w') as fc:
                            if "classification" in filepath:
                                fc.write("scientific_name,phylum,class,order,family,genus\n")
                            else:
                                fc.write("scientific_name,diet_type,activity_pattern,habitat_type,behavior_traits,average_lifespan\n")

def load_and_merge():
    print("Starting merge process...")
    
    # 1. Load Base Data
    if not os.path.exists(ANIMALS_JSON):
        print(f"Error: {ANIMALS_JSON} not found.")
        return

    with open(ANIMALS_JSON, 'r') as f:
        base_data = json.load(f)
    
    df_base = pd.DataFrame(base_data)
    print(f"Loaded {len(df_base)} base records.")

    # 2. Load External Data
    try:
        # Prefer XLSX if present, otherwise CSV
        if os.path.exists(CLASSIFICATION_XLSX):
            try:
                df_class = pd.read_excel(CLASSIFICATION_XLSX)
            except Exception:
                try:
                    df_class = pd.read_csv(CLASSIFICATION_CSV)
                except UnicodeDecodeError:
                    df_class = pd.read_csv(CLASSIFICATION_CSV, encoding='latin1')
        else:
            try:
                df_class = pd.read_csv(CLASSIFICATION_CSV)
            except UnicodeDecodeError:
                df_class = pd.read_csv(CLASSIFICATION_CSV, encoding='latin1')

        if os.path.exists(TRAITS_XLSX):
            try:
                df_traits = pd.read_excel(TRAITS_XLSX)
            except Exception:
                try:
                    df_traits = pd.read_csv(TRAITS_CSV)
                except UnicodeDecodeError:
                    df_traits = pd.read_csv(TRAITS_CSV, encoding='latin1')
        else:
            try:
                df_traits = pd.read_csv(TRAITS_CSV)
            except UnicodeDecodeError:
                df_traits = pd.read_csv(TRAITS_CSV, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # 3. Normalize External Data
    # Classification
    if 'Scientific.Name' in df_class.columns:
        df_class.rename(columns={
            'Scientific.Name': 'name',
            'Kingdom': 'kingdom',
            'Phylum': 'phylum',
            'Class': 'class',
            'Order': 'order',
            'Family': 'family',
            'Genus': 'genus'
        }, inplace=True)
    elif 'scientific_name' in df_class.columns:
        df_class.rename(columns={'scientific_name': 'name'}, inplace=True)
    
    # Traits
    if 'species' in df_traits.columns and 'body mass' in df_traits.columns:
        df_traits.rename(columns={
            'species': 'name',
            'body mass': 'body_mass',
            'metabolic rate': 'metabolic_rate',
            'brain size': 'brain_size'
        }, inplace=True)
    else:
        trait_mapping = {
            'scientific_name': 'name',
            'diet_type': 'diet',
            'habitat_type': 'habitat',
            'average_lifespan': 'lifespan',
            'behavior_traits': 'behavior'
        }
        df_traits.rename(columns=trait_mapping, inplace=True)

    # 4. Merge
    # Ensure name column is string and lower case for matching
    df_base['name_lower'] = df_base['name'].astype(str).str.lower()
    
    if df_class is not None and len(df_class) > 0:
        df_class['name_lower'] = df_class['name'].astype(str).str.lower()
        # Keep relevant columns
        cls_cols = ['name_lower', 'name', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        cls_cols = [c for c in cls_cols if c in df_class.columns]
        
        # Use outer join to include animals from external source that aren't in base
        df_base = pd.merge(df_base, df_class[cls_cols], on='name_lower', how='outer', suffixes=('', '_cls'))
        
        # Fill 'name' if it's missing (new record from class)
        if 'name_cls' in df_base.columns:
            df_base['name'] = df_base['name'].fillna(df_base['name_cls'])
            df_base.drop(columns=['name_cls'], inplace=True)

    if df_traits is not None and len(df_traits) > 0:
        df_traits['name_lower'] = df_traits['name'].astype(str).str.lower()
        # Keep relevant columns
        trt_cols = ['name_lower', 'name', 'diet', 'habitat', 'lifespan', 'behavior', 'body_mass', 'metabolic_rate', 'brain_size']
        trt_cols = [c for c in trt_cols if c in df_traits.columns]
        
        # Use outer join
        df_base = pd.merge(df_base, df_traits[trt_cols], on='name_lower', how='outer', suffixes=('', '_trt'))
        
        # Fill 'name' if it's missing (new record from traits)
        if 'name_trt' in df_base.columns:
            df_base['name'] = df_base['name'].fillna(df_base['name_trt'])
            df_base.drop(columns=['name_trt'], inplace=True)

    # Post-merge cleanup for new records
    df_base['source'] = df_base['source'].fillna('External-CSV')
    df_base['description'] = df_base['description'].fillna('Imported from external database.')
    df_base['species'] = df_base['species'].fillna(df_base['name'])
    # Ensure list fields are lists
    df_base['traits'] = df_base['traits'].apply(lambda x: x if isinstance(x, list) else [])
    df_base['source_urls'] = df_base['source_urls'].apply(lambda x: x if isinstance(x, list) else [])

    # 5. Fill Missing Fields
    # Logic: If base field is null/empty, take from merged columns
    fill_cols = ['diet', 'habitat', 'lifespan']
    for col in fill_cols:
        ext_col = f"{col}_trt"
        if ext_col in df_base.columns:
            if col in df_base.columns:
                df_base[col] = df_base[col].fillna(df_base[ext_col])
                df_base[col] = df_base[col].replace('', pd.NA).fillna(df_base[ext_col])
            else:
                df_base[col] = df_base[ext_col]
            df_base.drop(columns=[ext_col], inplace=True)

    # Drop temp join column
    df_base.drop(columns=['name_lower'], inplace=True)

    # 5. Identify unmatched records
    missing_matches = []
    # Check for rows where classification and trait columns are missing
    for i, row in df_base.iterrows():
        has_class = any([col in df_base.columns and pd.notna(row[col]) for col in ['phylum', 'class', 'order', 'family', 'genus']])
        has_trait = any([col in df_base.columns and pd.notna(row[col]) for col in ['diet', 'habitat', 'lifespan', 'behavior']])
        if not has_class and not has_trait:
            name = row.get('name') or row.get('name_cls') or row.get('name_trt')
            missing_matches.append(name if pd.notna(name) else None)

    # Write missing matches to a log file
    missing_log = os.path.join(EXTERNAL_DIR, 'missing_matches.log')
    with open(missing_log, 'w') as ml:
        ml.write('\n'.join([str(n) for n in missing_matches if n]))
    print(f"Logged {len(missing_matches)} unmatched names to {missing_log}")

    # 6. Clean and Save
    final_records = df_base.to_dict(orient='records')
    
    # Remove NaN values
    cleaned_data = []
    for record in final_records:
        clean_record = {}
        for k, v in record.items():
            if isinstance(v, list):
                clean_record[k] = v
            elif pd.notna(v) and v != "":
                clean_record[k] = v
        cleaned_data.append(clean_record)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"Successfully saved {len(cleaned_data)} enriched records to {OUTPUT_JSON}")

def main():
    ensure_directories()
    download_file(CLASSIFICATION_URL, CLASSIFICATION_CSV)
    download_file(TRAITS_URL, TRAITS_CSV)
    load_and_merge()

if __name__ == "__main__":
    main()
