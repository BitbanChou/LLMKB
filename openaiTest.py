

import os
import json
import time

import openai
from jsonargparse import CLI
from copy import deepcopy

import threading
from config import Config
from extractMovie import open_file

# replace the openai-key
openai.api_key = ''



# set organization
if os.environ.get('OPENAI_ORG') is not None:
    openai.organization = os.environ.get('OPENAI_ORG')

# set openai.api_key
if openai.api_key is None:
    raise Exception('OPENAI_API_KEY is not set')

def get_response(index, text, results, mess, temperature, max_tokens, model, EXSTING):
    try:
        #content = prompt.format(text)
        content = text
        if content in EXSTING:
            result = deepcopy(EXSTING[content])
            result['index'] = index
            print("Found in EXSTING!")

        else:
            resp = openai.ChatCompletion.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": content},
                ]
            )
            result = {'index': index, 'prompt': content, 'resp': resp['choices'][0]['message']['content']}
            EXSTING[content] = result

        results.append(result)

    except Exception as e:

        if e == KeyboardInterrupt:
            raise e
        print(e)
        time.sleep(2)
        results.append({'index': index, 'prompt': content,
                       'resp': "API Failed"})


def main(from_json: str = None, to_json: str = None,
         pretrained_model_name_or_path: str = 'gpt-3.5-turbo', temperature: float = 0.1, max_tokens: int = 1024,
         use_related_knowledge: int = 0, n_print: int = 100, n_samples: int = -1, input_field: str = 'input',
         use_tqdm=True,
         n_threads: int = 1,  existing_json: str = None):

    EXSTING = {}
    if existing_json is not None:
        with open(existing_json, 'r') as f:
            for l in f.readlines():
                d = json.loads(l)
                if d['resp'] != 'API Failed':
                    EXSTING[d['prompt']] = d

    with open(from_json, "r") as fr, open(to_json, 'w') as fw:

        threads, results = [], []

        lines = fr.readlines()
        total_lines = min(len(lines), n_samples) if n_samples > 0 else len(lines)
        start_time = time.time()

        for i, l in enumerate(lines):
            # if i>5:
            #     break
            if i == n_samples:
                break

            text = json.loads(l)[input_field]
            text = open_file('prompt.txt').replace('<<<<CONVERSATION>>>>', text).replace('<<<<QUESTION>>>>',
                                                                                         Config.prompt)
            execute_thread = threading.Thread(
                target=get_response,
                args=(i, text, results, '', temperature, max_tokens, pretrained_model_name_or_path, EXSTING)
            )
            threads.append(execute_thread)
            execute_thread.start()
            if len(threads) == n_threads:
                for execute_thread in threads:
                    execute_thread.join()

                for res in results:
                    fw.write(json.dumps(res)+'\n')

                threads = []
                results = []
                time.sleep(0)

            if i % n_print == 0:
                print(f'Time elapsed: {time.time() - start_time:.2f} sec. {i} / {total_lines} samples generated. ')

        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()

        for res in results:
            fw.write(json.dumps(res)+'\n')


if __name__ == '__main__':
    CLI(main)
