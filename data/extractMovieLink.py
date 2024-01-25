import json, re
import os
from jsonargparse import CLI

DIR = os.path.dirname(os.path.abspath(__file__))

all_set=[]
def main(dataset : str = None):
    with open(os.path.join(DIR,f'{dataset}/entity2id.json'),'r',encoding='utf-8') as file:
        dic = json.load(file)
        id2name = {v:k for k,v in dic.items()}

    with open(os.path.join(DIR,f'{dataset}/test.jsonl'),'r',encoding='utf-8') as fr:
        lines = fr.readlines()
        for i,l in enumerate(lines):
            entity_set=json.loads(l)['entity']
            for id in entity_set:
                if id not in all_set:
                    all_set.append(id)
                    with open(os.path.join(DIR,f'{dataset}/movieLink.txt'),'a',encoding='utf-8') as f:
                        f.write(id2name[id]+'\n')

if __name__=="__main__":
    CLI(main)