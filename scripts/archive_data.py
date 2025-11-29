import shutil
import os
from datetime import datetime

DATA_FILE = "data/animals.json"
VERSIONS_DIR = "data/versions"

def archive_data():
    if not os.path.exists(DATA_FILE):
        print(f"No data file found at {DATA_FILE}")
        return

    os.makedirs(VERSIONS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"animals_{timestamp}.json"
    archive_path = os.path.join(VERSIONS_DIR, archive_name)
    
    shutil.copy2(DATA_FILE, archive_path)
    print(f"Archived {DATA_FILE} to {archive_path}")

if __name__ == "__main__":
    archive_data()
