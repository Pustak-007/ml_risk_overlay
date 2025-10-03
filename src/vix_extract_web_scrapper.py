import requests
import re
import json
from pathlib import Path
import pandas as pd 
def fetch_modern_links():
    """Fetches all standard monthly VIX futures links from the main page."""
    page_url = "https://www.cboe.com/us/futures/market_statistics/historical_data/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(page_url, headers=headers, timeout=15)
        response.raise_for_status()
        html_content = response.text
        match = re.search(r"CTX\.defaultProductList = (\{.*?\});", html_content, re.DOTALL)
        if not match: return []
        data = json.loads(match.group(1))
        links = []
        base_cdn_url = "https://cdn.cboe.com/"
        monthly_pattern = re.compile(r'VX\+VXT/[A-Z]\d{1,2}$')
        for year_data in data.values():
            for product_info in year_data:
                if monthly_pattern.match(product_info.get("product_display", "")):
                    path = product_info.get("path")
                    if path: links.append(base_cdn_url + path)
        return links
    except Exception:
        return []

def fetch_archive_links():
    """Fetches all monthly VIX futures links from the archive page."""
    page_url = "https://www.cboe.com/us/futures/market_statistics/historical_data/archive/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(page_url, headers=headers, timeout=15)
        response.raise_for_status()
        html_content = response.text
        path_prefix_match = re.search(r"CTX\.historicalPathPrefix = '(.*?)';", html_content)
        product_list_match = re.search(r"CTX\.historicalProductLists = (\{.*?\});", html_content, re.DOTALL)
        if not path_prefix_match or not product_list_match: return []
        base_url = path_prefix_match.group(1)
        js_object_string = product_list_match.group(1)
        json_string = js_object_string.replace("'", '"').replace(',,', ',')
        cleaned_json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
        data = json.loads(cleaned_json_string)
        links = []
        vix_product_data = list(data.values())[0]
        for year_data in vix_product_data.values():
            for contract_info in year_data:
                path = contract_info.get("path")
                if path: links.append(f"{base_url}/{path}")
        return links
    except Exception:
        return []

def refine_and_combine_links(modern_links, archive_links):
    """Filters both lists by year and combines them in perfect chronological order."""
    
    filtered_archive = []
    for link in archive_links:
        match = re.search(r'CFE_[A-Z](\d{2})_VX\.csv$', link)
        if match:
            year = int("20" + match.group(1))
            if 2006 <= year <= 2013:
                filtered_archive.append(link)

    filtered_modern = []
    for link in modern_links:
        match = re.search(r'_(\d{4})-\d{2}-\d{2}\.csv$', link)
        if match:
            year = int(match.group(1))
            if 2014 <= year <= 2025:
                filtered_modern.append(link)
    
    # --- START OF THE FIX ---
    # Define a helper function to create a sort key (year, month) for archive files.
    def sort_key_for_archive(link):
        match = re.search(r'CFE_([A-Z])(\d{2})_VX\.csv$', link)
        if match:
            # Return a tuple: (year, month_code). Python sorts tuples element by element.
            return (match.group(2), match.group(1)) 
        return ('', '') # Default for safety

    # Sort the archive list using the custom key.
    filtered_archive.sort(key=sort_key_for_archive)
    
    # The modern list can still be sorted alphabetically because its format is YYYY-MM-DD.
    filtered_modern.sort()
    # --- END OF THE FIX ---
    
    combined_list = filtered_archive + filtered_modern
    return combined_list
def download_csv_files(url_list, destination_folder):
    """Downloads files from a list of URLs into a specified folder."""
    dest_path = Path(destination_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting download of {len(url_list)} files into '{dest_path}'...")
    
    for url in url_list:
        filename = url.split('/')[-1]
        file_path = dest_path / filename
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"  Successfully downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"  Failed to download {filename}. Error: {e}")

    print("\nDownload process finished.")

def read_csv_with_auto_skip(filepath):
    """
    Automatically detects and skips header text, loading only the data table.
    Works for files with or without disclaimer text.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_row_idx = None
    for i, line in enumerate(lines):
        # Look for common data header keywords
        if any(keyword in line for keyword in ['Trade Date', 'Date', 'Open', 'High', 'Low', 'Close']):
            # Check if this line looks like a header (has multiple column names)
            potential_cols = line.split()
            if len(potential_cols) >= 3:  # Likely a real header
                header_row_idx = i
                break
    
    if header_row_idx is None:
        return pd.read_csv(filepath)
    
    df = pd.read_csv(filepath, 
                     skiprows=header_row_idx,
                     engine='python')
    
    return df

if __name__ == "__main__":
    modern_links = fetch_modern_links()
    archive_links = fetch_archive_links()
    
    print(f"Initially found {len(modern_links)} modern links and {len(archive_links)} archive links.")
    refined_list = refine_and_combine_links(modern_links, archive_links)
    print(f"Found {len(refined_list)} relevant links after refining and combination.")
    for link in refined_list:
        print(link)