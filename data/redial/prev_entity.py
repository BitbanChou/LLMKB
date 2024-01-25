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

with open('entity2id.json', 'r', encoding='utf-8') as file:
    dic = json.load(file)
    name2id = {extract_movie_name(k): v for k, v in dic.items()}
    id2name = {v: k for k, v in name2id.items()}

with open('test.jsonl','r',encoding='utf-8') as fw:
    lines = fw.readlines()

def getMovieSet(text):
    movieSet = [id2name.get(item) for item in text]
    pre_arr = []
    for item in movieSet:
        if item is not None:
            pre_arr.append(item)
    return pre_arr

all_set={}
set=[]
for i,l in enumerate(lines):
    if(i<4000):
        all_set[i]=[]
        for item in getMovieSet(json.loads(l)['prev_entity']):
            all_set[i].append(item)

with open('relatedDic.json','w',encoding='utf-8') as fw:
    json.dump(all_set, fw, ensure_ascii=False)