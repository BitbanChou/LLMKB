import json, re
import os
from jsonargparse import CLI

DIR = os.path.dirname(os.path.abspath(__file__))

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

# def work(sentence,id2relation,name2id,dictionary,id2name):
#     movieSet = sentence
#     entity_set = []
#     for k, v in name2id.items():
#         if k in movieSet:
#             entity_set.append((k, v))
#     entity_set = sorted(entity_set, key=lambda x: len(x[0]))
#     #print(entity_set)
#     prompt_set=[]
#     for i in range(0,len(entity_set)):
#         #print(entity_set[i])
#         for k,v in dictionary.items():
#             #print(k)
#             if(int(k)==entity_set[i][1]):
#                 #print("123")
#                 for sub in v:
#                     if(sub[0]!=12):
#                         #print(id2name.get(int(k)))
#                         if id2name.get(sub[1]) is not None and id2name.get(int(k)) is not None:
#                             prompt_set.append("["+extract_movie_name(id2name.get(int(k))) + "," + id2relation.get(sub[0]) + "," + extract_movie_name(id2name.get(sub[1]))+"]")
#                             #print(id2name.get(int(k)),id2relation.get(sub[0]),extract_movie_name(id2name.get(sub[1]))
#             for sub in v:
#                 if sub[1]==entity_set[i][1] and sub[0]!=12 and id2name.get(sub[1]) is not None and id2name.get(int(k)) is not None:
#                     prompt_set.append("["+extract_movie_name(id2name.get(int(k)))+","+id2relation.get(sub[0])+","+extract_movie_name(id2name.get(sub[1]))+"]")
#                     #print(extract_movie_name(id2name.get(int(k))),id2relation.get(sub[0]),extract_movie_name(id2name.get(sub[1])))
#
#     prompt=','.join(elem for elem in prompt_set)
#     return prompt

def work(prev_entity,rec_entity,id2relation,name2id,dictionary,id2name):
    #print(prev_entity,rec_entity)
    prev_entity_set,rec_entity_set = [],[]
    for k, v in name2id.items():
        if k in rec_entity:
            rec_entity_set.append((k, v))
    entity_set = sorted(rec_entity_set, key=lambda x: len(x[0]))

    for k, v in name2id.items():
        if k in prev_entity:
            prev_entity_set.append((k, v))
    #print(prev_entity_set,entity_set)
    prompt_set=[]
    for i in range(0,len(entity_set)):
        for k,v in dictionary.items():
            if(int(k)==entity_set[i][1]):
                for sub in v:
                    if sub[0] != len(id2relation)-1:
                        if id2name.get(sub[1]) is not None and id2name.get(int(k)) is not None:
                            prompt_set.append("[" + extract_movie_name(id2name.get(int(k))) + "," + id2relation.get(
                                sub[0]) + "," + extract_movie_name(id2name.get(sub[1])) + "]")

            for sub in v: #
                if sub[1]==entity_set[i][1] and sub[0]!=len(id2relation)-1 and id2name.get(sub[1]) is not None and id2name.get(int(k)) is not None:
                    prompt_set.append("["+extract_movie_name(id2name.get(int(k)))+","+id2relation.get(sub[0])+","+extract_movie_name(id2name.get(sub[1]))+"]")

    prompt=','.join(elem for elem in prompt_set)
    return prompt_set

# def get_prompt(text,id2relation,name2id,dictionary,id2name):
#     text = [id2name.get(item) for item in text]
#     return work(text,id2relation,name2id,dictionary,id2name)

def get_prompt(text,text2,id2relation,name2id,dictionary,id2name):
    text = [id2name.get(item) for item in text]
    text2 = [id2name.get(item) for item in text2]
    return work(text,text2,id2relation,name2id,dictionary,id2name)

def extractInfo(movie,sentence,dictionary):
    arr=[]
    for k,v in dictionary.items():
        for attr in v:
            if attr.lower() in sentence.lower() and (k,attr) not in arr:
                arr.append((k,attr))

    res=[]
    for item in arr:
        relation,tail_entity=item[0],item[1]
        res.append([movie,relation,tail_entity])

    return res

def get_pre_entity(moveieSet,dictionary):
    arr=[]
    for k, v in dictionary.items():
        for attr in v:
            if attr.lower() in str(moveieSet).lower() and (k, attr) not in arr:
                arr.append((k, attr))

    return arr #','.join(elem for elem in moveieSet)

mp={'theater':'theatre','documentaries':'Documentary','christmas movies':'Christmas by medium'
    ,'santa clause movie':'Christmas by medium','horror or suspense':'Horror fiction',
    'horror and creepy':'Horror fiction','romances':'Romance novel','romance,':'Romance novel',
    'family related':'Family drama','crime,':'Crime film','sci-fi':'Science fiction'}
def get_movie_set(text,mappingDic):
    for k,v in mp.items():
        pos = text.lower().find(k.lower())
        if pos != -1 :
            text+=v

    arr=[]
    for k, v in mappingDic.items():
        for attr in v:
            if attr.lower() in text.lower() and (k, attr) not in arr:
                arr.append((k, attr))

    return arr

all_set=[]
def main(dataset : str = None):
    with open(os.path.join(DIR, f'{dataset}/entity2id.json'), 'r', encoding='utf-8') as file:
        dic = json.load(file)
        name2id = {extract_movie_name(k): v for k, v in dic.items()}
        id2name = {v: k for k, v in name2id.items()}

    with open(os.path.join(DIR, f'{dataset}/relation2id.json'), 'r', encoding='utf-8') as file:
        dic = json.load(file)
        id2relation = {v: extract_movie_name(k) for k, v in dic.items()}

    with open(os.path.join(DIR,f'{dataset}/dbpedia_subkg.json'), 'r', encoding='utf-8') as file:
        dictionary = json.load(file)

    with open(os.path.join(DIR,f'{dataset}/spiderMovieInfo.txt'), 'r', encoding='utf-8') as infile:
         spiderInfo = infile.readlines()

    with open(os.path.join(DIR,f'{dataset}/mappingDic.json'), 'r', encoding='utf-8') as file:
        mappingDic = json.load(file)

    with open(os.path.join(DIR,f'{dataset}/test.jsonl'), "r") as fr:
        lines = fr.readlines()
        for i, l in enumerate(lines):
            if i<4000:
                #print(i,end=": ")
                text = json.loads(l)['input']
                text2 = json.loads(l)['rec']
                text3 = json.loads(l)['prev_entity']
                moveieSet = get_movie_set(text,mappingDic)
                movieSet2 = [id2name.get(item) for item in text2]
                movieSet3 = [id2name.get(item) for item in text3]
                for movie in movieSet2:
                    for des in spiderInfo:
                        #print(movie,des)
                        if movie is not None and movie in des:
                            res=str(movie)+": "
                            # pre_arr=[]
                            # for item in movieSet3:
                            #     if item is not None:
                            #         pre_arr.append(["related", item])
                            # res+=str(pre_arr)
                            graphInfo=get_prompt(text, text2, id2relation, name2id, dictionary, id2name)
                            for item in extractInfo(movie,des,mappingDic):
                                if len(res)>500 :
                                    break
                                if str(item) not in graphInfo:
                                    graphInfo.append(str(item))

                            res += ','.join(str([elem[0],elem[1]]) for elem in moveieSet)
                            res += ','.join("[" + ",".join(elem.split(",")[1:]) for elem in graphInfo)
                            # print(movieSet2,end=": ")
                            # print(moveieSet)
                            # print(get_prompt(text2,id2relation,name2id,dictionary,id2name),end="")
                            # print(get_prompt(text,id2relation,name2id,dictionary,id2name))
                            with open(os.path.join(DIR,f'{dataset}/extractMovie.txt'), "a",encoding="utf-8") as fw:
                                if res[:500] not in all_set:
                                    all_set.append(res[:500])
                                    fw.write(res[:500]+'\n')
                                    # for line in spiderInfo:
                                    #     for item in movieSet2:
                                    #         if item is not None and item in line:
                                    #             fw.write(line)
                                    #             break
                        #relation_retrieval_query = open_file('prompt.txt').replace('<<<<MESSAGE>>>>', text).replace('<<<<CONVERSATION>>>>',getMovies(text)).replace('<<<<QUESTION>>>>', Config.prompt)

if __name__=="__main__":
    CLI(main)