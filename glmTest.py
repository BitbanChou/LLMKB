from transformers import AutoTokenizer, AutoModel
from jsonargparse import CLI
import time, json
from tqdm import tqdm
import torch

from fastchat.model.model_adapter import get_conversation_template
from easydict import EasyDict as edict
from extractMovie import open_file, get_pre_entity, get_related_entity
from config import Config

# please replace the path of GLM
model_name = "../../LLM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True,device_map="auto").half()

def get_glm_response(
    model,
    tokenizer,
    index,
    text,
    prompt,
    temperature=0.0,
    max_tokens=1024,
    results=None,
    args=None
):
    conv = get_conversation_template(args.model_path).copy()
    conv.append_message(conv.roles[0], prompt.format(text))
    conv.append_message(conv.roles[1], None)
    vicuna_input = conv.get_prompt()
    inputs = tokenizer([vicuna_input])
    output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).to(model.device),
            temperature=temperature,
            max_new_tokens=max_tokens,
    )
    # output_ids = model.stream_chat(torch.as_tensor(inputs.input_ids).to(model.device),
    #         temperature=temperature,
    #         max_new_tokens=max_tokens,query=inputs)
    #print(output_ids)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    #skip_echo_len = compute_skip_echo_len(args.model_path, conv, prompt)
    outputs = outputs[len(vicuna_input):]

    response = {'index': index, 'prompt': vicuna_input, 'resp': outputs}
    if results is not None:
        results.append(response)
    print(response)
    return response

def main(from_json: str = None, to_json: str = None,
         pretrained_model_name_or_path: str = 'gpt-3.5-turbo', temperature: float = 0.1, max_tokens: int = 1024,
         use_related_knowledge: int = 0, n_print: int = 100, n_samples: int = -1, input_field: str = 'input',
         use_tqdm=True,
         n_threads: int = 1, huggingface_model_type='AutoModelForCausalLM'):


    args = edict({
        'model_path': 'THUDM/chatglm2-6b',
        'device': 'cuda',
        'num_gpus': 1,
        'max_gpu_memory': '44Gib',
        'load_8bit': False,
        'debug': False,
    })

    with open(from_json, "r", encoding='utf-8') as fr, open(to_json, 'w') as fw:

        threads, results = [], []

        lines = fr.readlines()
        total_lines = len(lines)
        start_time = time.time()

        for i, l in tqdm(enumerate(lines), total=len(lines), disable=not use_tqdm):
            if i == n_samples:
                break
            text = json.loads(l)[input_field]
            text = open_file('prompt.txt').replace('<<<<CONVERSATION>>>>',text).replace('<<<<QUESTION>>>>',Config.prompt)
            get_glm_response(
                index=i,
                text=text,
                prompt=text,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                max_tokens=max_tokens,
                results=results,
                args=args
            )

            if i % n_print == 0:
                print(f'Time elapsed: {time.time() - start_time:.2f} sec. {i} / {total_lines} samples generated. ')

        for res in results:
            fw.write(json.dumps(res) + '\n')


if __name__=='__main__':
    CLI(main)
