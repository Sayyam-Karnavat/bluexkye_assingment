import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin
from collections import deque
import os
from urllib.parse import urlparse



def crawl_website(start_url : str, max_pages: int = 10)-> list[dict]:
    """
    Crawl a website starting from start_url, extract meaningful text, and save to JSON.
    """
    visited_urls = set()
    queue = deque([start_url])
    extracted_data = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    while queue and len(visited_urls) < max_pages:
        url = queue.popleft()
        if url in visited_urls:
            continue

        try:
            # Fetch the webpage
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            visited_urls.add(url)

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements (nav, footer, scripts, ads, etc.)
            for element in soup(['nav', 'footer', 'script', 'style', 'aside', 'iframe']):
                element.decompose()

            # Extract meaningful text (e.g., from p, h1-h6, article tags)
            meaningful_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article'])
            text_content = []
            for tag in meaningful_tags:
                text = tag.get_text(strip=True)
                if text and len(text) > 20:  # Ignore short snippets (e.g., menu items)
                    text_content.append(text)

            # Clean and combine text
            cleaned_text = ' '.join(text_content)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

            if cleaned_text:
                extracted_data.append({
                    'url': url,
                    'content': cleaned_text
                })

            # Find links to other pages on the same domain
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == urlparse(start_url).netloc and full_url not in visited_urls:
                    queue.append(full_url)

        except requests.RequestException as e:
            print(f"Error crawling {url}: {e}")
            continue

    return extracted_data

def save_to_json( data : list[dict] , dir : str = "Extracted_Data" ,  filename : str ='extracted_content.json'):
    """
    Save extracted data to a JSON file.
    """
    
    os.makedirs(dir , exist_ok=True)

    full_dir = os.path.join(dir , filename)
    # Used encoding utf-8 because of erros to parse special characters
    with open(full_dir, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return full_dir


if __name__ == "__main__":
    start_url = "https://dev.algorand.co/getting-started/introduction/" 
    extracted_data = crawl_website(start_url)
    save_to_json(extracted_data)
    print(f"Extracted content from {len(extracted_data)} pages and saved to extracted_content.json")