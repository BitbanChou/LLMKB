from jsonargparse import CLI
import time, json
from tqdm import tqdm
from extractMovie import open_file, get_pre_entity, get_related_entity
from config import Config
from document import DocumentService
from llm import LLMService
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import openai
import threading
from copy import deepcopy

# replace the openai-api_key
openai.api_key = ""
openai_api_key = openai.api_key

emd = HuggingFaceEmbeddings(model_name=Config.embedding_model_name)

class LangChainApplication(object):

    def __init__(self, model):
        self.config = Config
        self.llm_service = LLMService()

        print("load llm model ")
        self.llm_service.load_model(model_name_or_path=model)
        self.doc_service = DocumentService()
        print("load documents")
        self.doc_service.load_vector_store()

    def get_knowledge_based_answer(self, index, query, mess, dataset, chat_history=[],
                                   history_len=5,
                                   temperature=0.1,
                                   top_p=0.9, max_token=1024,
                                   top_k=3):
        content = query
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []
        self.llm_service.temperature = temperature
        self.llm_service.top_p = top_p
        self.llm_service.max_token = max_token


        chain = load_qa_chain(self.llm_service)
        message = content[content.find('conversation:'):content.find('.Your Task')]
        if mess != "":
            mess = get_related_entity(index,message,dataset)
        docs = self.doc_service.vector_store.similarity_search(mess + get_pre_entity(message,dataset),k=top_k)

        result = chain.run(input_documents=docs, question=content)
        #print(docs)

        response = {'index': index, 'prompt': content, 'resp': result}
        return response

def load_vector_store():
    return FAISS.load_local(Config.vector_store_path, emd)

def get_response(index, text, results, mess, dataset, max_tokens, top_k, EXSTING):
    try:
        content = text#prompt.format(text)

        if content in EXSTING:
            result = deepcopy(EXSTING[content])
            result['index'] = index
            print("Found in EXSTING!")

        else:
            vector_store = load_vector_store()
            chain = load_qa_chain(ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key,max_tokens=max_tokens,temperature=0.1))
            message = content[content.find('conversation:'):content.find('.Your Task')]
            if mess != "":
                mess = get_related_entity(index, message, dataset)
            docs = vector_store.similarity_search(mess + get_pre_entity(message,dataset),k=top_k)
            #get_related_entity(index,message)+
            result = chain.run(input_documents=docs, question=content)
            result = {'index': index, 'prompt': content, 'resp': result}
            EXSTING[content] = result
            #print(result)
        results.append(result)

    except Exception as e:

        if e == KeyboardInterrupt:
            raise e
        print(e)
        time.sleep(2)
        results.append({'index': index, 'prompt': content,
                       'resp': "API Failed"})


def main(from_json: str = None, to_json: str = None,
         pretrained_model_name_or_path: str = 'llama', temperature: float = 0.1, max_tokens: int = 1024,
         use_related_knowledge: int = 0, n_print: int = 100, n_samples: int = -1, input_field: str = 'input',
         use_tqdm=True,
         n_threads: int = 1, huggingface_model_type='AutoModelForCausalLM'):
    dataset = 'inspired'
    if 'redial' in from_json:
        dataset = 'redial'

    EXSTING = {}
    if pretrained_model_name_or_path != "gpt-3.5-turbo":
        application = LangChainApplication(pretrained_model_name_or_path)
    with open(from_json, "r", encoding='utf-8') as fr, open(to_json, 'w') as fw:

        threads, results = [], []

        lines = fr.readlines()
        total_lines = len(lines)
        start_time = time.time()
        arr = []
        for i, l in tqdm(enumerate(lines), total=len(lines), disable=not use_tqdm):
            # if i == n_samples:
            #     break
            text = json.loads(l)[input_field]

            if use_related_knowledge == 0:
                message = ""
            else:
                message = 'related_entity'
            text = open_file('prompt.txt').replace('<<<<CONVERSATION>>>>',text).replace('<<<<QUESTION>>>>',Config.prompt)
            if pretrained_model_name_or_path != "gpt-3.5-turbo":
                res = application.get_knowledge_based_answer(i, text, message, dataset, temperature=temperature,
                                                         max_token=max_tokens)
                results.append(res)
                if len(results) == 5:
                    for res in results:
                        fw.write(json.dumps(res) + '\n')

                    results = []
                    time.sleep(0)
            else:
                execute_thread = threading.Thread(
                    target=get_response,
                    args=(i, text, results, message, dataset, max_tokens, 3, EXSTING)
                )
                threads.append(execute_thread)
                execute_thread.start()
                if len(threads) == n_threads:
                    for execute_thread in threads:
                        execute_thread.join()

                    for res in results:
                        fw.write(json.dumps(res) + '\n')

                    threads = []
                    results = []
                    time.sleep(0)


            if i % n_print == 0:
                print(f'Time elapsed: {time.time() - start_time:.2f} sec. {i} / {total_lines} samples generated. ')

        for res in results:
            fw.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    CLI(main)
