# LLMKB: Large Language Models with Knowledge Base Augmentation for Conversational Recommendation

![intro](img/model.jpg)

This is the source code of KBs construction, LLMs scripts, and experimental results from our paper.


## Dataset

Please download DBpedia from the [link](https://databus.dbpedia.org/dbpedia/mappings/mappingbased-objects/2021.09.01/mappingbased-objects_lang=en.ttl.bz2), after unzipping, move it into `data`.

### Testing Data for Our Paper

We upload the test data in `data/` folder, which can be used to form the prompt to query different LLMs. The `inspired` and `redial` datasets are adapted from the data provided by [`CRSLab`](https://github.com/RUCAIBox/CRSLab/tree/main), where we added some additional data fields like `is_user`.

| File Name | Description | Example |
| -- | -- | -- |
|  `entity2id.json`  |  The mapping from the movie names (in `DBPedia` or `IMDB` format) to item ids. |  `{"<http://dbpedia.org/resource/Hoffa_(film)>": 0}`  |
|  `item_ids.json`  | A list of all the item ids.   |  `[0, 2049, 16388, 12292, 6, 4109, ...]` |
|  `test.json`  |  Conversational context (in `input` field) and target recommendation item (in `rec` field).  This type corresponds to c0 in the evaluation result. |  `{"input": "System: How did you like Hustlers? It definitely has the drama aspect...\n User: I liked it ..." "rec": [9722]}`  |
|  `test_p2.json`  |  Similar to `test.json`, but we use historical mentioned movies only.  This type corresponds to c1 in the evaluation result. |  `{"input": "System: Hustlers\n User:" "rec": [9722]}`  |
|  `test_p3.json`  |  Similar to `test.json`, but we use historical conversational text without mentioned movies. This type corresponds to c2 in the evaluation result. |  `{"input": "System: How did you like ? It definitely has the drama aspect...\n User: I liked it ..." "rec": [9722]}`  |
|  `test_raw.json`  |  Raw data file similar to the files provided in [`CRSLab`](https://github.com/RUCAIBox/CRSLab/tree/main)  |  `{"context": ["", ... "the last movie i saw in the theater was Hustlers . I generally like comedy, drama and documentaries"], "resp": "How did you like Hustlers? ...", "rec": [9722], "entity": [9722, 15748], "prev_entity": [15748, 17158, 8683, 8881, 16785, 9722], "dialog_id": "test_0", "turn_id": "test_4", "is_user": 0, "entity_name": ["drama", "Hustlers"]}`  |

## Construction of KBs proposed in our framework.

##### You can use the KBs we have built in the "resource/" folder for each dataset and place one of them in the "resource/faiss" folder.

### 1.Extract Structural Knowledge from each dataset.
```shell script
cd data 
# inspired
python extract_subkg.py --dataset inspired 

# redial
python extract_subkg.py --dataset redial
```
### 2.Obtain dbpedia links from `entity2id.json` and crawl the abstract information(Texual Knowledge) from all urls.
```shell script
# inspired
python extractMovieLink.py --dataset inspired
python spiderMovieInfo.py --dataset inspired

# redial
python extractMovieLink.py --dataset redial
python spiderMovieInfo.py --dataset redial
```
### 3.Obtain Entity Linking Knowledge from each dataset.
```shell script
# inspired
cd inspired
python prev_entity.py
cd ../

# redial
cd redial
python prev_entity.py
cd ../
```
### 3.Generate Mapping Dictionary for each dataset.
```shell script
# inspired
python extractMappingDictionary.py --dataset inspired

# redial
python extractMappingDictionary.py --dataset redial
```
### 4.Process heterogeneous knowledge in preparation for building a knowledge base.
```shell script
# inspired
python extractGraph.py --dataset inspired --type db # [db,db+related]

# redial
python extractGraph.py --dataset redial --type db # [db,db+related]
```
### 5.Place the generated "res.txt" from the previous step into the "resource/txt" directory.
```shell script
python document.py
```
After running the above code, the corresponding knowledge base will be generated in the "resource/faiss" directory.



## LLMs Scripts

To examine the impact of different types of knowledge base, we divide them into the following types:

• 𝑟𝑎𝑤 : results without using a KB.
• 𝑑𝑏 : results on loading a KB.
• 𝑑𝑏 + 𝑟𝑒𝑙𝑎𝑡𝑒𝑑 : results on loading a KB that contains entity linking knowledge

### 1.Install related dependencies
```shell script
pip install -r requirements.txt
```
### 2.Create config YAML file. Here an example from src/gpt-3.5/c0/inspired_config.yaml
    ```bash
    cat src/gpt-3.5/c0/inspired_config.yaml
    ```

    ```yaml
    from_json: ../data/inspired/test.jsonl # dataset path
    max_tokens: 1024
    use_related_knowledge: 0  # if use db+related please replace with 1
    n_threads: 1
    n_samples: -1
    pretrained_model_name_or_path: gpt-3.5-turbo # model path
    to_json: E:\pythonFiles\files\LLM+kg\src\gpt-3.5\c0\inspired_test.jsonl 
    ```
### 3.Run LLMs scripts with the YAML file:
#### For the 𝑟𝑎𝑤 setting:

```shell script
# GPT-3.5-t 
# please enter your openai-api_key before running
python openaiTest.py --config E:\pythonFiles\files\LLM+kg\src\llama\c0\inspired_config.yaml

# LLaMA
python llamaTest.py --config E:\pythonFiles\files\LLM+kg\src\llama\c0\inspired_config.yaml

# ChatGLM
python glmTest.py --config E:\pythonFiles\files\LLM+kg\src\glm\c0\inspired_config.yaml
```

#### For 𝑑𝑏 and 𝑑𝑏 + 𝑟𝑒𝑙𝑎𝑡𝑒𝑑 settings:
```shell script 
# if you use GPT-3.5-t, please enter your openai-api_key before running
python llmkb.py --config E:\pythonFiles\files\LLM+kg\src\glm\c0\inspired_config.yaml
```


### Experimental Results

We put all the experimental results in the src folder for checking.

### 1. Post-process your results in `${YOUR_DIR}`, here is an example for settings of c0,raw:
* Note:
dataset = [inspired,redial]
type = [raw,db,db+related]
For GPT-3.5-t,you should put the generated ${DATA}.json file in post-fix folder first for extraction
    ```bash
    # if this extraction is not special
    cp gpt-3.5/c0/extract.py ${YOUR_DIR}
    python ${YOUR_DIR}/extract.py --dataset inspired --type raw

    ```
#### After that a 'extracted.json' will generate in {type}/{dataset} folder.

### 2.For evaluation of recommendation task:
```shell script
    python tools/evaluate.py \
        --from_json ${YOUR_DIR}/c0/${DATA}/extracted.jsonl
```
### 3.For evaluation of conversation task:
```shell script
    python tools/evaluate_conv.py \
        --from_json ${YOUR_DIR}/c0/${DATA}/extracted.jsonl
```
