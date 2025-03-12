from mistralai import Mistral
import requests
from bs4 import BeautifulSoup
import csv

API_KEY = 'ICFcAmqJuKgEpH0f7ZjNJdCvuda9TV90'

def fetch_webpage_content(url):

    response = requests.get(url)
    if response.status_code == 200:

        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    else:

        raise Exception(f"Failed to retrieve webpage: {response.status_code}")

def get_response(client, url):

    content = fetch_webpage_content(url)

    # Initialize Mistral client
    model = "mistral-large-latest"

    system_content = """
        
        You are tasked with analyzing the main content of the article and analyzing every sentence.
        For every sentence, detect whether there is a subject, predicate and object, and output it
        in this format: (subject, predicate, object).
    """
    response = client.chat.complete(
        
        model=model,
        messages=[
            
            {"role": "system", "content": system_content},
            {"role": "user", "content": content},
        ],
        stream=False
    )

    response_content = response.choices[0].message.content

    return response_content


if __name__ == "__main__":

    client = Mistral(api_key=API_KEY)
    responses = []
    with open('globenewswire_articles_finance.csv', mode='r', newline='', encoding='utf-8') as file:
        
        reader = csv.DictReader(file)
        for row in reader:
            
            url = row.get('url')
            if url: responses.append(get_response(client, url))
            else: break

            break

    print(responses)