import json
import sys
import os

def validate(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        sys.exit(1)
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Root element is not a list.")
            sys.exit(1)
            
        if len(data) == 0:
            print("Error: Data list is empty.")
            sys.exit(1)
            
        # Check first item structure
        first = data[0]
        required_keys = ["name", "description"]
        for key in required_keys:
            if key not in first:
                print(f"Error: Missing required key '{key}' in first record.")
                sys.exit(1)
                
        print(f"Validation successful. Loaded {len(data)} records.")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate(sys.argv[1])
    else:
        validate("data/animals.json")
