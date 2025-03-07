import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from langdetect import detect

BASE_URL = "https://www.globenewswire.com"
NEWSROOM_URL = f"{BASE_URL}/newsroom"
MAX_ARTICLES = 100
RETRY_LIMIT = 1  # Max retries for failed requests
Industry = {
    'healtcare': f'{BASE_URL}/en/search/industry/Biotechnology,Pharmaceuticals%252520&%252520Biotechnology,Health%252520Care',
    'finance': f'{BASE_URL}/en/search/industry/Financials,Financial%252520Services,Financial%252520Administration',
    'tech': f'{BASE_URL}/en/search/industry/Software,Software%252520&%252520Computer%252520Services,Technology,Technology%252520Hardware%252520&%252520Equipment,Computer%252520Services,Computer%252520Hardware',
    # 'newsroom': NEWSROOM_URL
}
global_link_set = set()

def get_article_links(url, page_limit=10):
    """Extracts article links from the newsroom page."""
    links = set()  # Use a set to avoid duplicates
    for page in range(1, page_limit + 1):
        try:
            response = requests.get(f"{url}?page={page}&pageSize=12", timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        articles_divs = soup.find_all("div", class_="pagging-list-item-text-container")

        for div in articles_divs:
            for a in div.find_all("a"):
                link = a.get('href')
                if link and link.startswith("/news-release") and (link not in global_link_set):
                    links.add(BASE_URL + link)
                    global_link_set.add(BASE_URL + link)

        print(f"Extracted {len(articles_divs)} articles from page {page}")

        if len(links) >= MAX_ARTICLES:
            break
    return list(links)[:MAX_ARTICLES]

def scrape_article(url):
    """Extracts title, date, and content from an article with retry logic."""
    for attempt in range(RETRY_LIMIT):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.find("h1", class_="article-headline").text.strip()
            date = soup.find("time").text.strip()
            content_div = soup.find("div", class_="main-body-container article-body")
            content = " ".join([p.text.strip() for p in content_div.find_all("p")]) if content_div else "No content found"
            if(detect_language(content) == 'en'):
                return {"title": title, "date": date, "content": content, "url": url}
            return None

        except (requests.RequestException, AttributeError) as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
            # time.sleep(random.uniform(1, 3))  # Delay before retry

    print(f"Skipping article after {RETRY_LIMIT} failed attempts: {url}")
    return None

def detect_language(input):
    try:
        lang = detect(input)
        return lang
    except Exception as e:
        return 'gibberish'

def main():
    """Main function to scrape articles using multi-threading and save to CSV."""
    for industry_name, url in Industry.items():
        article_links = get_article_links(url)
        print(f"Total articles to scrape: {len(article_links)}")

        results = []
        for l in article_links:
            results.append(scrape_article(l))

        articles_data = []
        for article in results:
            if article:
                articles_data.append(article)

        df = pd.DataFrame(articles_data)
        df.to_csv(f"globenewswire_articles_{industry_name}.csv", index=False, encoding="utf-8")
    print("Scraping complete. Data saved to 'globenewswire_articles.csv'.")

if __name__ == "__main__":
    main()
