import json, re


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
    text = text.replace('_', ' ').replace('-', ' ').replace('>', ' ').replace(' PersonFunction 1', '')
    return del_space(del_parentheses(text))

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

mp={'theater':'theatre','documentaries':'Documentary','christmas movies':'Christmas by medium'
    ,'santa clause movie':'Christmas by medium','horror or suspense':'Horror fiction',
    'horror and creepy':'Horror fiction','romances':'Romance novel','romance,':'Romance novel',
    'family related':'Family drama','crime,':'Crime film','sci-fi':'Science fiction'}
def get_pre_entity(text,dataset):
    for k,v in mp.items():
        pos = text.lower().find(k.lower())
        if pos != -1 :
            text+=v

    with open(f'data/{dataset}/mappingDic.json', 'r', encoding='utf-8') as file:
        dictionary = json.load(file)
        arr=[]
        for k, v in dictionary.items():
            for attr in v:
                if attr.lower() in text.lower() and (k, attr) not in arr and attr.lower() != "eve":
                    arr.append((k, attr))

    return str(arr) #','.join(elem for elem in moveieSet)

def get_related_entity(i,text,dataset):
    with open(f'data/{dataset}/relatedDic.json', 'r', encoding='utf-8') as file:
        relatedDic = json.load(file)
    pre_arr = []
    for item in relatedDic[str(i)]:
        if item.lower() in text.lower():
            pre_arr.append(["related", item])

    return str(pre_arr)

all_set=[]
q=[]

