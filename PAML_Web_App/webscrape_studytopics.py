import requests
from bs4 import BeautifulSoup

spare_subtopics = ["General Concepts", "Fundamentals" , "Advanced" , "Work on examples" , "professor notes"]

def get_wikipedia_subtopics(topic):
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print("Error fetching Wikipedia page:", e)
        return spare_subtopics

    soup = BeautifulSoup(response.text, 'html.parser')
    subtopics = []
    
    toc_items = soup.find_all('li', class_='vector-toc-list-item')

    for item in toc_items:
        div = item.find('div', class_='vector-toc-text')
        if div:
            spans = div.find_all('span')
            if len(spans) >= 2:
                heading = spans[1].text.strip()
                heading = heading.replace("In ","").strip()
                
                if heading.lower() not in ['references', 'external links', 'see also', 'notes', 'bibliography' , 'sources', 'citations' , 'further reading']:
                    subtopics.append(heading)

    if len(subtopics) == 0:
        return spare_subtopics
    return subtopics


# if __name__ == "__main__":
#
#     topic = "Statistics"
#     subtopics = get_wikipedia_subtopics(topic)
#
#     print(f"Subtopics for '{topic}':")
#     for i, st in enumerate(subtopics, 1):
#         print(f"{i}. {st}")