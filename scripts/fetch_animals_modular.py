import json
import logging
import time
import random
import os
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import modal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_FILE = "data/animals.json"
MAX_ANIMALS_PER_RUN = 50 # Per source, or total? User said "limit to 50 items per run" generally. I'll limit total new items.

# Modal Configuration
app = modal.App("animal-scraper-modular")
image = modal.Image.debian_slim() \
    .pip_install("playwright", "beautifulsoup4") \
    .run_commands("playwright install-deps chromium", "playwright install chromium")

# --- Helper Functions ---

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def load_existing_data() -> List[Dict[str, Any]]:
    if not os.path.exists(OUTPUT_FILE):
        return []
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Ensure source_urls exists
        for entry in data:
            if "source" in entry:
                if "source_urls" not in entry:
                    entry["source_urls"] = [entry["source"]]
                del entry["source"]
            if "source_urls" not in entry:
                entry["source_urls"] = []
        return data
    except Exception as e:
        logger.error(f"Error loading existing data: {e}")
        return []

def save_data(data: List[Dict[str, Any]]):
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(data)} animals to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def merge_animal(existing: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merges new data into existing animal record."""
    # Merge source_urls
    existing_urls = set(existing.get("source_urls", []))
    new_urls = set(new_data.get("source_urls", []))
    existing["source_urls"] = list(existing_urls.union(new_urls))
    
    # Fill missing fields
    for key, value in new_data.items():
        if key not in existing or not existing[key]:
            existing[key] = value
        elif key == "traits" and isinstance(value, dict):
            if "traits" not in existing or not isinstance(existing["traits"], dict):
                existing["traits"] = {}
            # Merge traits
            for t_key, t_val in value.items():
                if t_key not in existing["traits"]:
                    existing["traits"][t_key] = t_val
    
    return existing

# --- Scraper Classes ---

class BaseScraper(ABC):
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url

    @abstractmethod
    def get_animal_links(self) -> List[str]:
        pass

    @abstractmethod
    def scrape_animal_details(self, url: str) -> Optional[Dict[str, Any]]:
        pass

# --- A-Z Animals Scraper ---

@app.function(image=image)
def az_get_links() -> List[str]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
    
    url = "https://a-z-animals.com/animals/"
    links = []
    try:
        print(f"Fetching A-Z Animals list from {url}...")
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--disable-blink-features=AutomationControlled'])
            context = browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            page = context.new_page()
            page.goto(url, timeout=60000)
            try:
                page.wait_for_selector("a[href*='/animals/']", timeout=10000)
            except:
                pass
            content = page.content()
            browser.close()
            
        soup = BeautifulSoup(content, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if '/animals/' in href and href.count('/') == 5:
                if href not in links and not href.endswith('/animals/') and "animals-that-start-with" not in href:
                    links.append(href)
        return list(set(links))
    except Exception as e:
        print(f"Error in az_get_links: {e}")
        return []

@app.function(image=image)
def az_scrape_details(url: str) -> Optional[Dict[str, Any]]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--disable-blink-features=AutomationControlled'])
            context = browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            page = context.new_page()
            page.goto(url, timeout=60000)
            content = page.content()
            browser.close()

        soup = BeautifulSoup(content, 'html.parser')
        
        # Name
        name_tag = soup.find('h1')
        name = name_tag.get_text(strip=True) if name_tag else ""
        
        # Description
        description = ""
        article = soup.find('article')
        if article:
            for p in article.find_all('p', recursive=False):
                text = p.get_text(strip=True)
                if text:
                    description = text
                    break
        
        # Traits
        traits = {}
        info_box = soup.find('div', {'class': 'animal-facts-box'})
        if info_box:
            for row in info_box.find_all('li'):
                key_tag = row.find('strong')
                if key_tag:
                    key = key_tag.get_text(strip=True).replace(':', '')
                    value = row.get_text(strip=True).replace(f"{key}:", "").strip()
                    traits[key] = value

        # Species (Scientific Name)
        species = traits.get("Scientific Name", "")
        
        return {
            "name": name,
            "species": species,
            "description": description,
            "traits": traits,
            "source_urls": [url]
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# --- Animal Corner Scraper ---

@app.function(image=image)
def ac_get_links() -> List[str]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
    
    url = "https://animalcorner.org/animals/"
    links = []
    try:
        print(f"Fetching Animal Corner list from {url}...")
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--disable-blink-features=AutomationControlled'])
            context = browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            page = context.new_page()
            page.goto(url, timeout=60000)
            content = page.content()
            browser.close()
            
        soup = BeautifulSoup(content, 'html.parser')
        # Links are like https://animalcorner.org/animals/aardvark/
        # They seem to be in a list.
        # Based on fetch_webpage, there is an "A-Z ANIMALS" section or just the main content.
        # Let's look for all links under /animals/ that are not the index itself.
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if 'animalcorner.org/animals/' in href:
                # Filter out non-animal pages if possible. 
                # The structure is usually /animals/name/
                # Avoid /animals/page/2/ etc if pagination exists (it seems to be all on one page or alphabetical)
                if href != url and not href.endswith('/animals/') and '/page/' not in href:
                     links.append(href)
        
        return list(set(links))
    except Exception as e:
        print(f"Error in ac_get_links: {e}")
        return []

@app.function(image=image)
def ac_scrape_details(url: str) -> Optional[Dict[str, Any]]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--disable-blink-features=AutomationControlled'])
            context = browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            page = context.new_page()
            page.goto(url, timeout=60000)
            content = page.content()
            browser.close()

        soup = BeautifulSoup(content, 'html.parser')
        
        # Name
        name_tag = soup.find('h1')
        name = name_tag.get_text(strip=True) if name_tag else ""
        # Clean name (remove "– ...", ": ...")
        name = re.split(r'[–:]', name)[0].strip()
        
        # Description - First paragraph after H1 usually
        description = ""
        # Skip generic site text
        skip_phrases = ["Discover the many amazing animals", "This post may contain affiliate links"]
        
        # Try to find the content div
        content_div = soup.find('div', {'class': 'entry-content'})
        if content_div:
            for p in content_div.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50 and "Kingdom:" not in text and not any(s in text for s in skip_phrases):
                    description = text
                    break
        
        if not description:
             for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50 and "Kingdom:" not in text and not any(s in text for s in skip_phrases):
                    description = text
                    break
        
        # Traits / Classification
        traits = {}
        # Look for the table with "Kingdom:", "Phylum:", etc.
        # Based on fetch_webpage: | Kingdom: | Animalia |
        for table in soup.find_all('table'):
            if "Kingdom" in table.get_text():
                for row in table.find_all('tr'):
                    cols = row.find_all(['td', 'th'])
                    if len(cols) == 2:
                        key = cols[0].get_text(strip=True).replace(':', '')
                        val = cols[1].get_text(strip=True)
                        traits[key] = val
        
        species = traits.get("Scientific Name", traits.get("Binomial name", ""))
        
        return {
            "name": name,
            "species": species,
            "description": description,
            "traits": traits,
            "source_urls": [url]
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# --- National Zoo Scraper ---

@app.function(image=image)
def nz_get_links() -> List[str]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
    
    base_url = "https://nationalzoo.si.edu/animals/list"
    links = []
    try:
        print(f"Fetching National Zoo list from {base_url}...")
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--disable-blink-features=AutomationControlled'])
            context = browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            page = context.new_page()
            
            # Scrape first 3 pages (approx 36 animals)
            for i in range(3):
                url = f"{base_url}?page={i}"
                print(f"Fetching {url}...")
                page.goto(url, timeout=60000)
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Links are like /animals/name
                # Look for links inside the main content area
                # Based on fetch_webpage: [American flamingo](https://nationalzoo.si.edu/animals/american-flamingo)
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if '/animals/' in href and href.count('/') == 2: # /animals/name
                        full_url = f"https://nationalzoo.si.edu{href}" if href.startswith('/') else href
                        if full_url not in links and "list" not in full_url and "animals" != full_url.split('/')[-1]:
                             links.append(full_url)
                
            browser.close()
        
        return list(set(links))
    except Exception as e:
        print(f"Error in nz_get_links: {e}")
        return []

@app.function(image=image)
def nz_scrape_details(url: str) -> Optional[Dict[str, Any]]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--disable-blink-features=AutomationControlled'])
            context = browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            page = context.new_page()
            page.goto(url, timeout=60000)
            content = page.content()
            browser.close()

        soup = BeautifulSoup(content, 'html.parser')
        
        # Name
        name_tag = soup.find('h1')
        name = name_tag.get_text(strip=True) if name_tag else ""
        
        # Description
        description = ""
        # Skip generic site text
        skip_phrases = ["See the Smithsonian's National Zoo's Giant Pandas", "Sign up for our newsletter"]
        
        # Usually in a div with class 'body-text' or similar, or just paragraphs
        # Let's look for the first substantial paragraph
        intro = soup.find('div', {'class': 'intro-text'})
        if intro:
             text = intro.get_text(strip=True)
             if not any(s in text for s in skip_phrases):
                 description = text
        
        if not description:
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50 and "Kingdom:" not in text and not any(s in text for s in skip_phrases):
                    description = text
                    break
        
        # Traits (Physical Description, etc.)
        traits = {}
        # National Zoo has "Physical Description", "Food/Eating Habits", etc. as H3s or H4s
        # Let's try to grab some headers and their content
        for header in soup.find_all(['h3', 'h4']):
            header_text = header.get_text(strip=True)
            if header_text in ["Physical Description", "Diet", "Habitat", "Lifespan"]:
                # Get next sibling or next paragraph
                content_sib = header.find_next_sibling(['p', 'div'])
                if content_sib:
                    traits[header_text] = content_sib.get_text(strip=True)

        species = traits.get("Scientific Name", "") # Might not be explicitly labeled as such
        
        return {
            "name": name,
            "species": species,
            "description": description,
            "traits": traits,
            "source_urls": [url]
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# --- Main Execution ---

@app.local_entrypoint()
def main():
    existing_data = load_existing_data()
    existing_map = {normalize_text(d["name"]).lower(): d for d in existing_data}
    
    # 1. A-Z Animals
    az_links = az_get_links.remote()
    print(f"A-Z Animals: Found {len(az_links)} links.")
    
    # 2. Animal Corner
    ac_links = ac_get_links.remote()
    print(f"Animal Corner: Found {len(ac_links)} links.")
    
    # 3. National Zoo
    nz_links = nz_get_links.remote()
    print(f"National Zoo: Found {len(nz_links)} links.")
    
    # Combine and prioritize
    
    def get_name_from_url(url):
        if "a-z-animals.com" in url:
            return url.strip('/').split('/')[-1].replace('-', ' ')
        elif "animalcorner.org" in url:
            return url.strip('/').split('/')[-1].replace('-', ' ')
        elif "nationalzoo.si.edu" in url:
            return url.strip('/').split('/')[-1].replace('-', ' ')
        return ""

    # Filter A-Z links (we probably already have some)
    new_az_links = []
    for link in az_links:
        name_guess = get_name_from_url(link)
        if normalize_text(name_guess).lower() not in existing_map:
            new_az_links.append(link)
            
    # Filter Animal Corner links
    new_ac_links = []
    enrich_ac_links = []
    
    for link in ac_links:
        name_guess = get_name_from_url(link)
        if normalize_text(name_guess).lower() not in existing_map:
            new_ac_links.append(link)
        else:
            enrich_ac_links.append(link)
            
    # Filter National Zoo links
    new_nz_links = []
    enrich_nz_links = []
    
    for link in nz_links:
        name_guess = get_name_from_url(link)
        if normalize_text(name_guess).lower() not in existing_map:
            new_nz_links.append(link)
        else:
            enrich_nz_links.append(link)

    print(f"Potential new animals from A-Z: {len(az_links)}") # Simplified
    print(f"Potential new animals from Animal Corner: {len(new_ac_links)}")
    print(f"Potential new animals from National Zoo: {len(new_nz_links)}")

    tasks = []
    
    # Add enrichment tasks (prioritize NZ and AC)
    for link in enrich_nz_links[:10]:
        tasks.append(("nz", link))
    for link in enrich_ac_links[:10]:
        tasks.append(("ac", link))
        
    # Add new tasks
    remaining_slots = MAX_ANIMALS_PER_RUN - len(tasks)
    if remaining_slots > 0:
        # Mix of sources
        combined_new = [("nz", l) for l in new_nz_links] + [("ac", l) for l in new_ac_links]
        # Add some A-Z if needed, but let's focus on new sources
        if len(combined_new) < remaining_slots:
             # Add A-Z links that are likely new
             new_az = [l for l in az_links if get_name_from_url(l) not in existing_map]
             combined_new.extend([("az", l) for l in new_az])
             
        random.shuffle(combined_new)
        tasks.extend(combined_new[:remaining_slots])
        
    print(f"Selected {len(tasks)} tasks for this run.")
    
    # Execute
    results = []
    for source, url in tasks:
        if source == "az":
            res = az_scrape_details.remote(url)
        elif source == "ac":
            res = ac_scrape_details.remote(url)
        elif source == "nz":
            res = nz_scrape_details.remote(url)
        else:
            res = None
            
        if res:
            results.append(res)
            
    # Merge and Save
    for res in results:
        name_key = normalize_text(res["name"]).lower()
        if name_key in existing_map:
            print(f"Merging data for {res['name']}")
            existing_map[name_key] = merge_animal(existing_map[name_key], res)
        else:
            print(f"Adding new animal: {res['name']}")
            existing_map[name_key] = res
            
    save_data(list(existing_map.values()))

if __name__ == "__main__":
    # This part is for local testing if needed, but Modal runs via `modal run`
    pass
