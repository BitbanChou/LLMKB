from bs4 import BeautifulSoup as bs
import requests
import time
import dotenv
import os

dotenv.load_dotenv()

base_url = 'https://www.imdb.com/find/'
headers = {
    'Accept': '*/*',
    'User-Agent': 'Mozilla/5.0'
}

file_filmes = open('filmList.txt', 'r', encoding='utf-8').readlines()
id_filmes = []


def limparId(href):
    return href.split('/')[2]

real_movie_count = 0
i = 1
for filme in file_filmes:
    if len(filme) < 1:
        continue
    results = requests.get(
        url=base_url,
        headers=headers,
        params={'q': filme}
    )

    soup = bs(results.content, 'html.parser')
    lista_filmes = soup.find('ul', class_='ipc-metadata-list')

    if lista_filmes is None:
        print('Film Not Found.')
        continue
    else:
        real_movie_count+=1
    filme = lista_filmes.find('li')
    titulo = filme.find('a')
    ano = filme.find('span')
    if (titulo is not None) and (ano is not None):
        id_filmes.append(limparId(titulo['href']))
        #print(f'{i:03d}/{len(file_filmes)} | {titulo.text} - {ano.text} - {limparId(titulo["href"])}')

    time.sleep(.5)
    i += 1
    print(real_movie_count)

file_ids = open('ids.txt', 'w').write('\n'.join(id_filmes))
