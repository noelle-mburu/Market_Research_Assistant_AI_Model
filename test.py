import requests
import json
from duckduckgo_search import DDGS
from docling.document_converter import DocumentConverter

def search_web(query,max_results = 20):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query,max_results=max_results):
            results.append(r["href"])
    return results

def extract_text(url):
    try:
        converter = DocumentConverter()
        results = converter.convert(url)
        markdown = results.document.export_to_markdown()
        print(markdown)
        return markdown
    except Exception as e:
        print(f"Failed to extract url: {e}")
        return " "
    
#run a pipeline
if __name__ == "__main__":
    query = "Roblox gaming market research statistics https://sqmagazine.co.uk/roblox-statistics/ https://sqmagazine.co.uk/gen-z-gaming-platform-preferences-statistics/"

    print("Starting extractions")

    urls = search_web(query, max_results=10)

    all_data = []

    for url in urls:
        print(f"Extracting from {url} ...")
        text = extract_text(url)

        all_data.append({"url": url, "content": text[:1000]})  # store first 1000 chars
        

    # Save results to JSON for your AI agent
    
    with open("extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("\nData extraction complete. Saved to extracted_data.json")


    