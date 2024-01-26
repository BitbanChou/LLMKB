import re

import json
from jsonargparse import CLI
from collections import Counter
from editdistance import eval as distance
import numpy as np
import pandas as pd
def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)


def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()


def del_numbering(text):
    pattern = r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?"
    return re.sub(pattern, "", text)


def is_in(text, items, threshold):
    for i in items:
        if (distance(i.lower(), text.lower()) <= threshold):
            return True
    return False


def nearest(text, items):
    """ given the raw text name and all candidates,
        return {movie_name:, min_edit_distance: , nearest_movie: }
    """
    # calculate the edit distance
    items = list(set(items))
    dists = [distance(text.lower(), i.lower()) for i in items]
    # find the nearest movie
    nearest_idx = np.argmin(dists)
    nearest_movie = items[nearest_idx]
    return {
        'movie_name': text,
        'min_edit_distance': dists[nearest_idx],
        'nearest_movie': nearest_movie
    }

def extract_movie_name(text):
    text = text.split('/')[-1]
    text = text.replace('_', ' ').replace('-', ' ').replace('>', ' ')
    return del_space(del_parentheses(text))


def extract_list(l, candidates=None):
    if(type(l['resp'])!=str):
        text = l['resp']['choices'][0]['message']['content']
    else:
        text = l['resp']
    print(text)
    # try:
    #     preference, text = text.split('1.', maxsplit=1)
    # except Exception as e:
    #     print(e)
    #     preference = ""
    #     text = text.replace(',', '\n')
    # rec_list = [del_numbering(del_space(del_parentheses(i.strip()))) for i in text.split('\n')]
    # if candidates is not None:
    #     rec_list = [nearest(i, candidates) for i in rec_list]
    # l['rec_list'] = rec_list
    # l['preference'] = preference
    return l

import re

def is_not_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    #print(re.compile(pattern).findall(text))
    return len(re.compile(pattern).findall(text)) <1

#E:\\LLMs-as-Zero-Shot-Conversational-RecSys-master\\src\\gpt-4\\general\\intermediate\\redial\\
def main(from_json:str='E:\\pythonFiles\\files\\LLM+kg\\src\\gpt-3.5\\c0\\db\\inspired\\extracted.jsonl'):
    #with open(from_json, encoding='utf-8') as file:
    preds = [json.loads(l) for l in open(from_json)]
    gts = [json.loads(l)['gt_list'] for l in open(from_json)]
    gts_set = []
    gts_count = 0
    for arr in gts:
        for ele in arr:
            if ele not in gts_set:
                gts_set.append(ele)
    #print("--------")

    # gt
    array = []
    for line in gts:
        array.append(line)

    counter=Counter([item for sublist in array for item in sublist])

    sorted_counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
    gt_sum = sum(item[1] for item in sorted_counter)

    gts=[x[0] for x in sorted_counter]

    max_movie_length=0
    for movie in gts:
        if len(movie) > max_movie_length :
            max_movie_length = len(movie)

    rec_in_gt = []

    print("max_movie_length: ",max_movie_length)
    # rec
    array2 = []
    for i,l in enumerate(preds):
        # if i>5000:
        #     continue
        # if (type(l['resp']) != str):
        #     text = l['resp']['choices'][0]['message']['content']
        # else:
        #     text = l['resp']
        text=l['rec_list']
        text=[i['movie_name'] for i in text]
        rec_list = text#[del_numbering(del_space(del_parentheses(i.strip()))) for i in text.split('\n')]
        if rec_list[0] in gts_set :
            gts_count+=1
        print(rec_list[0])
        # for i in rec_list:
        #     array2.append(i)
        # for i in rec_list:
        #     if i in gts:
        if rec_list[0] in gts and rec_list[0] not in rec_in_gt:
            rec_in_gt.append(rec_list[0])

        if is_not_chinese(rec_list[0]) == True and len(rec_list[0])<max_movie_length:
            array2.append(rec_list[0])

        # for item in sorted_counter:
        #     if item[0] == "S" and rec_list[0] == "S":
        #         array2.append("S")
        #     else:
        #         if item[0] in rec_list[0]:
        #             array2.append(item[0])

    counter2=Counter(array2)

    sorted_counter2=sorted(counter2.items(),key=lambda x:x[1],reverse=True)
    print(sorted_counter2)
    rec_sum = sum(int(item[1]) for item in sorted_counter2)
    print("ground-truth count:", len(sorted_counter))
    print("rec items in gts count:", len(rec_in_gt))
    print("rec items count:", len(sorted_counter2))
    print("top-1 item in gts_set count:", gts_count)

    # for item in sorted_counter2:
    #     print(item[0])

    cnt=0
    res=[]
    long_tail=[]
    for item1 in sorted_counter:
        flag=0
        for item2 in sorted_counter2:
            if item1[0] == item2[0]:
                flag=1
                long_tail.append(item2[1])
                res.append([item1[0],100*item1[1]/gt_sum,100*item2[1]/rec_sum])
                if(100*item1[1]/gt_sum>100*item2[1]/rec_sum):
                    cnt+=1
                #print(item1[0],item1[1],item2[1])
        # if flag==0:
        #     res.append([item1[0], 100 * item1[1] / gt_sum, 0])

    # for i in sorted(long_tail,reverse=True):
    #     print(i)

    # res_all = pd.DataFrame(res)
    # res_all.to_csv('gpt4rec.csv', index=False)

if __name__ == '__main__':
    CLI(main)