import requests
from bs4 import BeautifulSoup
from razdel import sentenize
import pickle

category_urls = ['http://www.lib.ru/INPROZ/']
for start_url in category_urls:
    html_doc = requests.get(start_url)
    soup = BeautifulSoup(html_doc.content, 'html.parser')

    author_links = soup.find_all('a')
    for link in author_links[1:-1]:
        if '.' not in link['href'] and link['href'].count('/') == 1:
            #print(link['href'])

            author_html = requests.get(start_url+link['href'])
            book_soup = BeautifulSoup(author_html.content, 'html.parser')
            book_links = book_soup.find_all('a')
            for book_link in book_links:
                try:
                    if 'txt' in book_link['href'] and 'Contents' not in book_link['href']:
                        print('inter url:', book_link['href'])
                except:
                    continue
                    #print('error:', book_link)
                try:
                    if 'txt' in book_link['href'] and 'Contents' not in book_link['href']:
                        book_payload = []
                        text_url = start_url+link['href']+book_link['href']
                        text_html = requests.get(text_url)
                        print('book link:', start_url+link['href']+book_link['href'])
                        text_soup = BeautifulSoup(text_html.content, 'html.parser')
                        header = text_soup.find_all('h2')[0].text
                        print('header:', header)
                        text_parts = text_soup.find_all('pre')
                        text = text_parts[0].contents[1].text
                        sentences = list(sentenize(text))
                        for sentence in sentences:
                            #print('\n\n', sentence.text)
                            sent_for_search_prepared = sentence.text.replace(',', '%2C').replace(' ', '%20').replace('\n', '%0A').replace('-', '%2D')
                            sent_highlight_url = text_url + '#:~:text=' + sent_for_search_prepared
                            #print('url:', sent_highlight_url, '\n\n')
                            book_payload.append((sentence.text, sent_highlight_url, header))
                        with open(f'data/{book_link["href"][:-4]}_data.pkl', 'wb') as f:
                            pickle.dump(book_payload, f)
                except:
                    pass
