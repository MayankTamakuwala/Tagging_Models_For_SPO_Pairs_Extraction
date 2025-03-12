from mistralai import Mistral
import requests
from bs4 import BeautifulSoup
import csv
import re

API_KEY = 'ICFcAmqJuKgEpH0f7ZjNJdCvuda9TV90'

def fetch_webpage_content(url):

    response = requests.get(url)
    if response.status_code == 200:

        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    else:

        raise Exception(f"Failed to retrieve webpage: {response.status_code}")

# Use MistralAI to parse through each webpage
# and format the body of the article in the format
# (subject, predicate, object).
def get_response(client, url):

    # The content of the webpage that the
    # AI agent will operate on
    content = fetch_webpage_content(url)

    model = "mistral-large-latest"

    # Tells the AI agent its role and what it should do. 
    system_content = """
        
        You are tasked with analyzing the main content of the article and analyzing every sentence.
        For every sentence, detect whether there is a subject, predicate and object, and output it
        in this format: (subject, predicate, object).
    """

    # Use AI to create this judgement
    response = client.chat.complete(
        
        model=model,
        messages=[
            
            {"role": "system", "content": system_content},
            {"role": "user", "content": content},
        ],
        stream=False
    )

    # Retrieve results for analysis on current webpage.
    response_content = response.choices[0].message.content

    return response_content

# Parse responses to only retrieve objects that are (subject, predicate, object)
def get_sub_pred_obj(responses):

    triplets = []
    for r in responses:

        # Use regex to parse only the answers that the agent generated,
        # and not the other content that it generated
        pattern = r'\((.*?)\)'
        triple = re.findall(pattern, r)
        
        # Store each answer in triplets list after finding valid
        # subject, predicate, object triplets
        for t in triple:

            # Delimit each part by comma
            elements = [element.strip() for element in t.split(',')]
            if len(elements) == 3: triplets.append(tuple(elements))

    return triplets


if __name__ == "__main__":

    client = Mistral(api_key=API_KEY)
    responses = []
    with open('globenewswire_articles_finance.csv', mode='r', newline='', encoding='utf-8') as file:
        
        reader = csv.DictReader(file)
        for row in reader:
            
            url = row.get('url')
            if url: responses.append(get_response(client, url))
            else: break

            #break

    print(get_sub_pred_obj(responses))

    """

    Example output:

    [('Karolinska Development AB', 'announces', 'positive feedback'), ('PharmNovo', 'received', 'positive feedback'), ('The meeting', 'aimed', 'to provide guidance'), ('PharmNovo', 'conducted', 'a regulatory pre-IND Type B meeting'), ('Such meetings', 'are held', 'before submitting an IND application'), ('PharmNovo', 'presented', 'preclinical data'), ('PharmNovo', 'sought', 'advice'), ('PharmNovo', 'received', 'guidance'), ('FDA', 'did not direct', 'any negative remarks'), ('FDA', 'provided', 'useful guidance'), ('PharmNovo', 'plans', 'to apply for IND approval'), ('the company', 'aims', 'to apply for approval'), ('PN6047', 'is', 'a selective delta-opioid receptor agonist'), ('PN6047', 'being developed', 'as 
a new treatment'), ('Pain management', 'is', 'an area of significant commercial interest'), ('We', 'are pleased', 'with the outcome'), ('We', 'look forward', "to following the company's continued efforts"), ("Karolinska Development's ownership", 'amounts', 'to 20%'), ('Karolinska Development AB', 'is', 'a Nordic life sciences investment company'), ('The company', 'focuses', 'on identifying breakthrough medical innovations'), ('The company', 'invests', 'in the creation and growth of companies'), ('Karolinska Development', 'has access', 'to world-class medical innovations'), ('The Company', 'aims', 'to build companies'), ('Karolinska Development', 'has', 'a portfolio of eleven companies'), ('The company', 'is led', 'by an entrepreneurial team')]
    """