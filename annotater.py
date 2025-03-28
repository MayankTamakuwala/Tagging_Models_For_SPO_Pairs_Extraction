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
        You are a precise text analyzer focused on extracting subject-predicate-object triplets from text, with specific focus on named entities.
        Rules:
        1. Only extract complete triplets where all three elements are present
        2. Focus on factual statements and clear relationships
        3. Skip complex or ambiguous sentences
        4. Only include named entities as subjects and objects:
           - PEOPLE (e.g., "John Smith", "CEO", "researchers")
           - PLACES (e.g., "New York", "Europe", "headquarters")
           - ORGANIZATIONS (e.g., "Apple Inc.", "FDA", "research team")
        5. Exclude conceptual entities like:
           - Abstract concepts (e.g., "steady growth plan", "market strategy")
           - Generic terms (e.g., "the company", "the team")
           - Time periods (e.g., "next quarter", "last year")
           - Products or services (unless they are specific named products)
        6. Output only in the format: (subject, predicate, object)
        7. Keep subjects and objects as concise named entities
        8. Keep predicates as single verbs or short verb phrases
        9. Separate multiple triplets with newlines
        10. Do not provide any explanations or additional text
        11. If no clear triplets with named entities are found, return empty
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
    with open('globenewswire_articles_finance.csv', mode='r', newline='', encoding='utf-8') as infile, \
        open('finance_articles_triplets.csv', mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        writer.writerow(['url', 'triplets'])

        for row in reader:

            url = row.get('url')
            if not url: continue

            response = get_response(client, url)
            parsed_triplets = get_sub_pred_obj([response])
            triplet_strings = [f"({s}, {p}, {o})" for s, p, o in parsed_triplets]
            writer.writerow([url] + triplet_strings)

            ###
            #break

    """

    Example output:

    [('Karolinska Development AB', 'announces', 'positive feedback'), ('PharmNovo', 'received', 'positive feedback'), ('The meeting', 'aimed', 'to provide guidance'), ('PharmNovo', 'conducted', 'a regulatory pre-IND Type B meeting'), ('Such meetings', 'are held', 'before submitting an IND application'), ('PharmNovo', 'presented', 'preclinical data'), ('PharmNovo', 'sought', 'advice'), ('PharmNovo', 'received', 'guidance'), ('FDA', 'did not direct', 'any negative remarks'), ('FDA', 'provided', 'useful guidance'), ('PharmNovo', 'plans', 'to apply for IND approval'), ('the company', 'aims', 'to apply for approval'), ('PN6047', 'is', 'a selective delta-opioid receptor agonist'), ('PN6047', 'being developed', 'as 
a new treatment'), ('Pain management', 'is', 'an area of significant commercial interest'), ('We', 'are pleased', 'with the outcome'), ('We', 'look forward', "to following the company's continued efforts"), ("Karolinska Development's ownership", 'amounts', 'to 20%'), ('Karolinska Development AB', 'is', 'a Nordic life sciences investment company'), ('The company', 'focuses', 'on identifying breakthrough medical innovations'), ('The company', 'invests', 'in the creation and growth of companies'), ('Karolinska Development', 'has access', 'to world-class medical innovations'), ('The Company', 'aims', 'to build companies'), ('Karolinska Development', 'has', 'a portfolio of eleven companies'), ('The company', 'is led', 'by an entrepreneurial team')]

    After the changes, this is the output:
    [('PharmNovo', 'received', 'FDA feedback'), ('PharmNovo', 'presented', 'preclinical data'), ('PharmNovo', 'sought advice on', 'CMC activities'), ('PharmNovo', 'received guidance on', 'Phase 2a study design'), ('PharmNovo', 'plans to apply for', 'IND approval'), ('PharmNovo', 'aims to apply for', 'clinical Phase 2a trial approval'), ('Viktor Drvota', 'is', 'CEO'), ('Karolinska Development', 'has ownership in', 'PharmNovo'), ('Paavo Truu', 'introduced', 'special mortgage offer'), ('Coop Pank', 'offers', 'Rahasahtel savings account'), ('Coop Pank', 'offers', 'Lastehoius childrens savings account'), ('Coop Pank', 'became', 'second bank'), ('Coop Eesti', 'comprises', '320 stores'), ('YieldMax', 'announced', 'distributions'), ('YieldMax', 'is based in', 'CHICAGO'), ('YieldMax', 'is based in', 'MILWAUKEE'), ('YieldMax', 'is based in', 'NEW YORK'), ('Gavin Filmore', 'is associated with', 'Tidal Financial Group'), ('Tidal Financial Group', 'advises', 'YieldMax ETFs'), ('Foreside Fund Services', 'distributes', 'YieldMax ETFs'), ('Investec Bank plc', 'is', 'Joint Broker'), ('Investec Bank plc', 'made disclosures', 'De La Rue plc'), ('Priyali Bhattacharjee', 'is', 'Contact name'), ('PricewaterhouseCoopers', 'audited', 'EfTEN Real Estate Fund AS'), ('Supervisory Board', 'approved', 'annual report'), 
    ('Supervisory Board', 'submitted', 'profit distribution proposal'), ('Supervisory Board', 'proposed', 'approve the annual report 2024'), ('Supervisory Board', 'proposed', 'distribute the undistributed profit'), ('Supervisory Board', 'proposed', 'extend the authorisations'), ('Supervisory Board', 'proposed', 'delegate the increase of the share capital'), ('Supervisory Board', 'authorised', 'carry out all activities'), ('Viljar Arakas', 'appointed', 'representative'), ('Bitget', 'received', 'VASP'), ('Bitget', 'distributed', '70 million dollar'), ('Bitget', 'launched', 'Bitget Builders'), ('Bitget', 'supports', 'Bybit'), ('Bitget', 'integrated', 'Callpay'), ('Bitget', 'introduced', 'USDT'), ('Bitget', 'integrated', 'Abstract Mainnet'), ('Bitget', 'launched', 'Bitget Graduates'), ('Bitget', 'partners', 'LALIGA'), ('Bitget', 'partners', 'Buse Tosun Çavuşoğlu'), ('Bitget', 'partners', 'Samet Gümüş'), ('Bitget', 'partners', 'İlkin Aydın'), ('Alexandre Johnson', 'said', 'THSYU'), ('Jessica Green', 'is', 'Chief Operating Officer'), ('THSYU', 'has unveiled', 'security enhancements'), ('THSYU', 'is redefining', 'French market'), ('THSYU', 'is positioning', 'a leader'), ('THSYU', 'is providing', 'users'), ('THSYU', 'has also implemented', 'cloud-based infrastructure'), 
    ('THSYU', 'delivers', 'stable and secure experience'), ('THSYU', 'is raising', 'cryptocurrency exchange'), ('Industry analysts', 'predict', 'THSYUs bold advancements'), ('Mark Elliott', 'is', 'THRUVISION GROUP PLC'), ('NB Private Equity Partners', 'announced', 'Jefferies International Limited'), ('Luke Mason', 'contact', 'NBPE Investor Relations'), ('Charles Gorman', 'contact', 'Kaso Legg Communications'), ('Luke Dampier', 'contact', 'Kaso Legg Communications'), ('Charlotte Francis', 'contact', 'Kaso Legg Communications'), ('NB Alternatives Advisers', 'subsidiary', 'Neuberger Berman Group'), ('Neuberger Berman', 'founded', '1939'), ('Neuberger Berman', 'manages', '500 billion'), ('UNPRI', 'named', 'Neuberger Berman'), ('Pensions & Investments', 'named', 'Neuberger Berman'), ('NBPE', 'established', 'Guernsey'), ('NBPE', 'received', 'Guernsey Financial Services Commission'), ('HSBC Bank Plc', 'made disclosures in respect of', 'Learning Technologies Group plc'), ('Dhruti Singh', 'is the contact name for', 'HSBC Bank Plc'), ('TAG Associates', 'won', 'Best Due Diligence Processes'), ('David Basner', 'is', 'CEO of TAG'), ('Jonathan Bergman', 'is', 'President of TAG'), ('TAG investment team', 'conducts', '600 manager meetings'), ('PAM Awards', 'honor', 'achievements in the US private wealth management space'), 
    ('Awards dinner', 'took place', 'Guastavinos')]
    """