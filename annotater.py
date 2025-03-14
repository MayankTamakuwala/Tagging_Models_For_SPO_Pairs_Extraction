from mistralai import Mistral
import requests
from bs4 import BeautifulSoup
import csv
import re
from time import sleep
from typing import Optional

API_KEY = 'ICFcAmqJuKgEpH0f7ZjNJdCvuda9TV90'
MAX_RETRIES = 3
MAX_TOKENS = 4096  # Adjust based on your needs

def clean_text(text: str) -> str:
    """Clean and normalize text before processing."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    return text.strip()

def extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main article content while removing boilerplate."""
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()
    
    # Try to find main content using common article containers
    main_content = None
    for selector in ['article', 'main', '.article-content', '.post-content']:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    return main_content.get_text() if main_content else soup.get_text()

def fetch_webpage_content(url: str) -> Optional[str]:
    """Fetch and preprocess webpage content with error handling."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        content = extract_main_content(soup)
        return clean_text(content)
    
    except requests.RequestException as e:
        print(f"Error fetching {url}: {str(e)}")
        return None

def get_response(client: Mistral, url: str) -> Optional[str]:
    """Get AI response with error handling and retries."""
    content = fetch_webpage_content(url)
    if not content:
        return None

    model = "mistral-large-latest"
    system_content = """
        You are a precise text analyzer focused solely on extracting subject-predicate-object triplets from text.
        Rules:
        1. Only extract complete triplets where all three elements are present
        2. Focus on factual statements and clear relationships
        3. Skip complex or ambiguous sentences
        4. Ignore descriptive or narrative text that doesn't contain clear triplets
        5. Output only in the format: (subject, predicate, object)
        6. Keep subjects and objects as concise noun phrases
        7. Keep predicates as single verbs or short verb phrases
        8. Separate multiple triplets with newlines
        9. Do not provide any explanations or additional text
        10. If no clear triplets are found, return empty
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": content},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.1,  # Lower temperature for more focused outputs
                stream=False
            )
            return response.choices[0].message.content
        
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to process {url} after {MAX_RETRIES} attempts: {str(e)}")
                return None
            sleep(2 ** attempt)  # Exponential backoff

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

    After the changes, this is the output:
    [('PharmNovo', 'received', 'positive feedback'), ('PharmNovo', 'presented', 'preclinical data'), ('PharmNovo', 'sought', 'advice'), ('PharmNovo', 'received', 'guidance'), ('PharmNovo', 'plans to apply for', 'IND approval'), ('PharmNovo', 'aims to apply for', 'approval'), ('PharmNovo', 'is developing', 'PN6047'), ('PN6047', 'is', 'selective delta-opioid receptor agonist'), ('Karolinska Development', 'owns', '20 percent of PharmNovo'), ('customer base', 'grown by', '13'), ('customer deposits', 'increased by', '3 million euros'), ('customer deposits', 'reached', '1.93 billion euros'), ('corporate customer deposits', 'increased by', '15 million euros'), ('private customer deposits', 'increased by', '5 million euros'), 
    ('international platform deposits', 'decreased by', '17 million euros'), ('bank deposits', 'grown by', '12'), ('loan portfolio', 'increased by', '15 million euros'), ('loan portfolio', 'reached', '1.79 billion euros'), ('business loans', 'increased by', '8 million euros'), ('home loans', 'increased by', '7 million euros'), ('loan portfolio', 'grown by', '18'), ('loan impairment cost', 'was', '0.1 million euros'), ('net income', 'decreased by', '6'), ('expenses', 'increased by', '1'), ('net profit', 'earned', '2.3 million euros'), ('net profit', 'earned', '5.0 million euros'), ('return on equity', 'was', '13.7'), ('cost-income ratio', 'was', '52'), ('Coop Pank', 'achieved', 'solid business growth'), 
    ('Coop Pank', 'grew', 'domestic deposit volume'), ('Coop Pank', 'offers', 'Rahasahtel savings account'), ('Rahasahtel savings account', 'has', '2 interest rate'), ('Coop Pank', 'offers', 'Lastehoius childrens savings account'), ('Lastehoius childrens savings account', 'has', '3.1 interest rate'), ('Euribor', 'increased', 'interest in business and home loans'), ('Coop Pank', 'introduced', 'special mortgage offer'), ('Coop Pank', 'gave', 'customers the option to stop using physical bank cards'), ('Coop Pank', 'became', 'the second bank in Estonia to offer a more convenient and secure digital payment solution'), ('Coop Pank', 'continues on', 'steady growth path'), ('customer base', 'grown by', '13'), 
    ('deposits', 'grown by', '12'), ('Coop Pank', 'operates', 'in Estonia'), ('Coop Pank', 'aims', 'to bring everyday banking services closer to peoples homes'), ('bank', 'is', 'Coop Pank'), ('strategic shareholder', 'is', 'Coop Eesti'), ('Coop Eesti', 'comprises', '320 stores'), ('YieldMax', 'announced', 'distributions'), ('YieldMax', 'lists', 'ETFs'), ('YieldMax', 'holds', 'ETFs'), ('YieldMax', 'provides', 'distributions'), ('YieldMax', 'invests', 'options'), ('YieldMax', 'invests', 'ETFs'), ('YieldMax', 'invests', 'securities'), ('YieldMax', 'invests', 'assets'), ('YieldMax', 'invests', 'derivatives'), ('YieldMax', 'invests', 'contracts'), ('YieldMax', 'invests', 'receipts'),]
    """