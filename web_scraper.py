import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://www.globenewswire.com"
NEWSROOM_URL = f"{BASE_URL}/newsroom"
MAX_ARTICLES = 300
THREADS = 10  # Number of concurrent threads
RETRY_LIMIT = 1  # Max retries for failed requests

def get_article_links(page_limit=10):
    """Extracts article links from the newsroom page."""
    links = set()  # Use a set to avoid duplicates
    for page in range(1, page_limit + 1):
        url = f"{NEWSROOM_URL}?page={page}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        articles_divs = soup.find_all("div", class_="pagging-list-item-text-container")

        for div in articles_divs:
            for a in div.find_all("a"):
                link = a.get('href')
                if link and link.startswith("/news-release"):
                    links.add(BASE_URL + link)

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

            return {"title": title, "date": date, "content": content, "url": url}

        except (requests.RequestException, AttributeError) as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
            # time.sleep(random.uniform(1, 3))  # Delay before retry

    print(f"Skipping article after {RETRY_LIMIT} failed attempts: {url}")
    return None

def main():
    """Main function to scrape articles using multi-threading and save to CSV."""
    article_links = get_article_links()
    print(f"Total articles to scrape: {len(article_links)}")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        results = executor.map(scrape_article, article_links)

    articles_data = []
    for article in results:
        if article:
            articles_data.append(article)

    df = pd.DataFrame(articles_data)
    df.to_csv("globenewswire_articles.csv", index=False, encoding="utf-8")
    print("Scraping complete. Data saved to 'globenewswire_articles.csv'.")

if __name__ == "__main__":
    main()
