import os
import bz2
import re
import xml.etree.ElementTree as ET
import requests

# Configuration
URL = "https://dumps.wikimedia.org/swwiki/latest/swwiki-latest-pages-articles.xml.bz2"
ARCHIVE_PATH = "./data/swwiki-latest-pages-articles.xml.bz2"
OUTPUT_TXT_PATH = "./data/swwiki_articles.txt"

def download_dump():
    """Downloads the Swahili Wikipedia dump using streaming."""
    print("🚀 Downloading Swahili Wikipedia dump... This may take a few minutes.")
    
    # Custom headers to satisfy Wikimedia's User-Agent Policy
    headers = {
        "User-Agent": "MiniLLMApp/1.0 (contact: msodoki@parrot.local) PythonRequests/2.0"
    }
    
    # Added headers=headers to the GET request below
    response = requests.get(URL, headers=headers, stream=True)
  
    if response.status_code == 200:
        with open(ARCHIVE_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("✅ Download complete!")
    else:
        print(f"❌ Failed to download. HTTP Status code: {response.status_code}")
        exit()

def clean_wiki_text(text):
    """Removes basic Wikitext formatting to leave raw text."""
    if not text:
        return ""
    # Remove internal wiki links, keeping only the display text
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Strip out template references like {{Kigezo:...}}
    text = re.sub(r"\{\{[^\}]*\}\}", "", text)
    # Clean section header markers (e.g., == Historia ==)
    text = re.sub(r"==+\s*([^=]+)\s*==+", r"\1", text)
    return text.strip()

def process_xml_to_txt():
    """Parses compressed XML progressively to prevent memory overload."""
    print("⏳ Extracting text and converting to plain .txt format...")
    
    # Internal Swahili Wikipedia namespace prefixes to skip (templates, categories, files)
    skip_prefixes = ["Kigezo:", "Picha:", "Msaada:", "Wikipedia:", "Jamii:", "Majadiliano:", "MediaWiki:"]

    try:
        # Read directly from the .bz2 wrapper
        with bz2.BZ2File(ARCHIVE_PATH, "rb") as xml_file:
            context = ET.iterparse(xml_file, events=("end",))
            
            with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as txt_file:
                for _, elem in context:
                    # Look for page structures
                    if elem.tag.endswith("page"):
                        title_elem = elem.find("{*}title")
                        text_elem = elem.find("{*}revision/{*}text")
                        
                        if title_elem is not None and text_elem is not None:
                            title = title_elem.text
                            content = text_elem.text
                            
                            # Filter out system and meta-pages
                            if title and not any(title.startswith(p) for p in skip_prefixes):
                                cleaned = clean_wiki_text(content)
                                if cleaned:
                                    txt_file.write(f"=== {title} ===\n")
                                    txt_file.write(cleaned)
                                    txt_file.write("\n\n")
                        
                        # Free up memory immediately after processing a page
                        elem.clear()
                        
        print(f"🎉 Success! Cleaned text saved to: {OUTPUT_TXT_PATH}")
    except Exception as e:
        print(f"❌ An error occurred during parsing: {e}")

if __name__ == "__main__":
    # Only download if you don't already have the file locally
    if not os.path.exists(ARCHIVE_PATH):
        download_dump()
    else:
        print("ℹ️ Archive file already exists locally. Skipping download.")
        
    process_xml_to_txt()