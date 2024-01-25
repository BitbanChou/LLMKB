import json,os
import re
import requests
from bs4 import BeautifulSoup
from jsonargparse import CLI

DIR = os.path.dirname(os.path.abspath(__file__))

header = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
}

def getTopKSentence(sentence,topK):
    sentences=sentence.split('.')
    if(len(sentences)>topK):
        print('.'.join(sentences[:topK]))
        return '.'.join(sentences[:topK])
    else:
        print(sentence)
        return sentence

def spider(link,dataset):
    #print(link)
    url = link.replace('\n','').lstrip('<').rstrip('>') #"http://dbpedia.org/resource/Hustlers_(film)"
    resp = requests.get(url, headers=header)
    result=re.compile(r'"dbo:abstract" lang="en" >.*?<\/span>').findall(resp.text)
    if result:
        result = result[0]
        #print(result.lstrip('"dbo:abstract" lang="en" >').rstrip('<\/span>'))
        with open(os.path.join(DIR,f'{dataset}/spiderMovieInfo.txt'),'a',encoding='utf-8') as file:
            file.write(getTopKSentence(result.lstrip('"dbo:abstract" lang="en" >').rstrip('<\/span>'),2)+'\n')
    else:
        print("没有匹配到结果")
    #print(result)
    # if(len(result)>0):
def main(dataset : str = None):
    # spider
    with open(os.path.join(DIR,f'{dataset}/movieLink.txt'),'r',encoding='utf-8') as f:
        linkset=f.readlines()
        for i in range(len(linkset)):
            if(i>270):
                #print(link)
                spider(linkset[i],dataset)

if __name__ == '__main__':
    CLI(main)

    # 提取
    # with open('res.txt', 'r', encoding='utf-8') as f:
    #     doc = f.readlines()
    #     for text in doc:
    #         getTopKSentence(text,2)
    #        print(text[text.find('information:'):text.find('Your Task')])
    # 排序
    # with open('E:/pythonFiles/files/LLM+kg/src/gpt-3.5/general/inspired_test.jsonl', 'r', encoding='utf-8') as f:
    #     doc = f.readlines()
    #     arr=[]
    #     for line in doc:
    #         arr.append(line)
    #     idx=191
    #     while len(arr)>0:
    #         for item in arr:
    #             if(item.find("index\": "+str(idx))!=-1):
    #                 print(item,end="")
    #                 idx+=1
    #                 break