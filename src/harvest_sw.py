import os
import requests

# Target file configuration
OUTPUT_PATH = "./data/swwiki_10mb.txt"
TARGET_SIZE_BYTES = 10 * 1024 * 1024  # Exactly 10 Megabytes

# Wikimedia API Endpoint
API_URL = "https://wikipedia.org"

def get_random_articles():
    """Fetches a batch of random Swahili article titles."""
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 10
    }
    # Updated to look like a standard modern browser to prevent blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(API_URL, params=params, headers=headers)
        if response.status_code == 200:
            # Safely check if content type is JSON before parsing
            if "application/json" in response.headers.get("Content-Type", ""):
                data = response.json()
                return [item["title"] for item in data["query"]["random"]]
            else:
                print("⚠️ API did not return JSON. It returned HTML instead.")
        else:
            print(f"⚠️ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error fetching titles: {e}")
    return []

def get_article_text(title):
    """Fetches plain text content of a specific article."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title
    }
    # Match the browser headers here too
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(API_URL, params=params, headers=headers)
        if response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
            data = response.json()
            pages = data["query"]["pages"]
            for page_id in pages:
                return pages[page_id].get("extract", "")
    except Exception as e:
        pass
    return ""

def harvest_dataset():
    """Gathers text sequentially until reaching the target size."""
    current_size = 0
    
    if os.path.exists(OUTPUT_PATH):
        current_size = os.path.getsize(OUTPUT_PATH)
        print(f"ℹ️ Output file exists. Current size: {current_size / (1024*1024):.2f} MB")
    
    print(f"🎯 Target size: {TARGET_SIZE_BYTES / (1024*1024):.2f} MB")
    
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        while current_size < TARGET_SIZE_BYTES:
            # 1. Grab a list of random article titles
            titles = get_random_articles()
            
            for title in titles:
                # Double-check size boundary mid-loop
                if current_size >= TARGET_SIZE_BYTES:
                    break
                
                # 2. Get the clean text from the API
                text = get_article_text(title)
                
                if text.strip():
                    # 3. Append to dataset
                    f.write(text + "\n\n")
                    f.flush() # Force write to disk immediately
                    
                    # Update local metrics
                    current_size = os.path.getsize(OUTPUT_PATH)
                    progress = (current_size / TARGET_SIZE_BYTES) * 100
                    print(f"⏳ Progress: {progress:.2f}% | Current size: {current_size / (1024*1024):.2f} MB", end="\r")
                    
    print(f"\n\n✅ Done! Dataset successfully created at: {OUTPUT_PATH}")

if __name__ == "__main__":
    harvest_dataset()