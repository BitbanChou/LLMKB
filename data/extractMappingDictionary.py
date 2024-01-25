import os
from jsonargparse import CLI
DIR = os.path.dirname(os.path.abspath(__file__))

import json,re
def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)

def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()

def del_numbering(text):
    pattern = r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?"
    return re.sub(pattern, "", text)

def extract_movie_name(text):
    text = text.split('/')[-1]
    text = text.replace('_', ' ').replace('-', ' ').replace('>', ' ')
    return del_space(del_parentheses(text))

def main(dataset : str = None):
    with open(os.path.join(DIR, f'{dataset}/entity2id.json'), 'r', encoding='utf-8') as file:
        dic = json.load(file)
        id2name = {v: extract_movie_name(k) for k, v in dic.items()}

    with open(os.path.join(DIR, f'{dataset}/relation2id.json'), 'r', encoding='utf-8') as f:
        relation2id = json.load(f)
        id2relation = {v: k for k, v in relation2id.items()}

    with open(os.path.join(DIR, f'{dataset}/dbpedia_subkg.json'), 'r', encoding='utf-8') as f:
        sub_kg = json.load(f)

    mp=[[] for _ in range(len(relation2id))]
    for k,v in sub_kg.items():
        for sub in v:
            #print(sub[0],sub[1])
            if sub[0]!=len(relation2id)-1 and sub[1] not in mp[sub[0]]:
                mp[sub[0]].append(id2name.get(sub[1]))

    dic={}
    for i in range(len(relation2id)-1):
        dic[extract_movie_name(id2relation.get(i))]=mp[i]

    with open(os.path.join(DIR, f'{dataset}/mappingDic.json'), 'w', encoding='utf-8') as fw:
        json.dump(dic, fw, ensure_ascii=False)
#print(mp)
# for i in range(len(mp)):
#     print(id2relation.get(i),end=": ")
#     print(mp[i])

if __name__=="__main__":
    CLI(main)