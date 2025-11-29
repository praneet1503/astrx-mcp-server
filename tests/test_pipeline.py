import os
import json
import shutil
import subprocess
import time

def test_pipeline():
    print("--- Starting Pipeline Simulation ---")
    
    # 1. Simulate Scraper Output
    print("\n[1] Simulating Scraper Run...")
    dummy_data = [
        {"name": "Test Animal", "description": "This is a test animal.", "diet": "Bits", "habitat": "Server"},
        {"name": "Mock Bird", "description": "A bird that mocks.", "diet": "Bugs", "habitat": "Tree"}
    ]
    
    # Backup existing data if any
    if os.path.exists("data/animals.json"):
        shutil.copy("data/animals.json", "data/animals.json.bak")
        
    with open("data/animals.json", "w") as f:
        json.dump(dummy_data, f, indent=2)
    print("Generated dummy data/animals.json")
    
    # 2. Validate Data
    print("\n[2] Validating Data...")
    result = subprocess.run(["python3", "scripts/validate_data.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Validation Failed!")
        print(result.stderr)
        return
    else:
        print("Validation Passed.")

    # 3. Archive Data
    print("\n[3] Archiving Data...")
    result = subprocess.run(["python3", "scripts/archive_data.py"], capture_output=True, text=True)
    print(result.stdout)
    
    # Verify archive creation
    versions_dir = "data/versions"
    files = sorted(os.listdir(versions_dir))
    if files:
        latest = files[-1]
        print(f"Found archive: {latest}")
    else:
        print("Error: No archive found.")
        return

    # 4. Simulate Git Commit (Dry Run)
    print("\n[4] Simulating Git Commit...")
    print("git add data/animals.json data/versions/")
    print("git commit -m 'Auto-update animals.json'")
    print("git push")
    
    # Restore original data
    if os.path.exists("data/animals.json.bak"):
        shutil.move("data/animals.json.bak", "data/animals.json")
        print("\nRestored original data/animals.json")
    
    print("\n--- Pipeline Simulation Complete ---")

if __name__ == "__main__":
    test_pipeline()
