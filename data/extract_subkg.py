import json
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm
import os
import sys
from jsonargparse import CLI

DIR = os.path.dirname(os.path.abspath(__file__))


def get_item_set(file):
    entity = set()
    if "inspired" in file:
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)
                for turn in line:
                    for e in turn['movie_link']:
                        entity.add(e)
    elif "redial" in file:
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)
                for message in line['messages']:
                    for e in message['movie']:
                        entity.add(e)
    return entity

def load_kg(path):
    print('load kg')
    kg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            tuples = line.strip().split()
            if tuples and len(tuples) == 4 and tuples[-1] == ".":
                h, r, t = tuples[:3]
                kg[h].append((r, t))
    return kg


def extract_subkg(kg, seed_set, n_hop):
    """extract subkg from seed_set by n_hop

    Args:
        kg (dict): {head entity: [(relation, tail entity)]}
        seed_set (list or set): [entity]
        n_hop (int):

    Returns:
        subkg (dict): {head entity: [(relation, tail entity)]}, extended by n_hop
    """
    print('extract subkg')

    subkg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
    subkg_hrt = set()  # {(head_entity, relation, tail_entity)}

    ripple_set = None
    for hop in range(n_hop):
        memories_h = set()  # [head_entity]
        memories_r = set()  # [relation]
        memories_t = set()  # [tail_entity]

        if hop == 0:
            tails_of_last_hop = seed_set  # [entity]
        else:
            tails_of_last_hop = ripple_set[2]  # [tail_entity]

        for entity in tqdm(tails_of_last_hop):
            for relation_and_tail in kg[entity]:
                h, r, t = entity, relation_and_tail[0], relation_and_tail[1]
                if (h, r, t) not in subkg_hrt:
                    subkg_hrt.add((h, r, t))
                    subkg[h].append((r, t))
                memories_h.add(h)
                memories_r.add(r)
                memories_t.add(t)

        ripple_set = (memories_h, memories_r, memories_t)

    return subkg


# def kg2id(kg, dataset, all_entity):
#     entity_set = all_entity
#     with open(os.path.join(DIR, f'{dataset}/relation_set.json'), encoding='utf-8') as f:
#         relation_set = json.load(f)
#
#     with open(os.path.join(DIR, f'{dataset}/entity2id.json'), encoding='utf-8') as f:
#         entity2id = json.load(f)
#
#     temp_entity_set = set()
#     for k, v in entity2id.items():
#         if v in entity_set:
#             temp_entity_set.add(k)
#     entity_set = temp_entity_set
#     #print(entity_set)
#     for head, relation_tails in tqdm(kg.items()):
#         for relation_tail in relation_tails:
#             if relation_tail[0] in relation_set:
#                 entity_set.add(head)
#                 entity_set.add(relation_tail[1])
#
#     #entity2id = {e: i for i, e in enumerate(entity_set)}
#
#     print(f"# entity: {len(entity2id)}")
#     relation2id = {r: i for i, r in enumerate(relation_set)}
#     relation2id['self_loop'] = len(relation2id)
#     print(f"# relation: {len(relation2id)}")
#
#     kg_idx = {}
#     for head, relation_tails in kg.items():
#         if head in entity2id:
#             head = entity2id[head]
#             kg_idx[head] = [(relation2id['self_loop'], head)]
#             for relation_tail in relation_tails:
#                 if relation_tail[0] in relation2id and relation_tail[1] in entity2id:
#                     kg_idx[head].append((relation2id[relation_tail[0]], entity2id[relation_tail[1]]))
#
#     return entity2id,relation2id, kg_idx


def kg2id(kg, dataset, all_entity):
    entity_set = all_entity

    with open(os.path.join(DIR, f'{dataset}/relation_set.json'), encoding='utf-8') as f:
        relation_set = json.load(f)

    with open(os.path.join(DIR, f'{dataset}/entity2id.json'), encoding='utf-8') as f:
        entity2id = json.load(f)

    temp_entity_set = set()
    for k, v in entity2id.items():
        if v in entity_set:
            temp_entity_set.add(k)
    entity_set = temp_entity_set
    # print(entity_set)
    for head, relation_tails in tqdm(kg.items()):
        for relation_tail in relation_tails:
            if relation_tail[0] in relation_set:
                entity_set.add(head)
                entity_set.add(relation_tail[1])

    #entity2id = {e: i for i, e in enumerate(entity_set)}
    print(f"# entity: {len(entity2id)}")
    relation2id = {r: i for i, r in enumerate(relation_set)}
    relation2id['self_loop'] = len(relation2id)
    print(f"# relation: {len(relation2id)}")

    kg_idx = {}
    for head, relation_tails in kg.items():
        if head in entity2id:
            head = entity2id[head]
            kg_idx[head] = [(relation2id['self_loop'], head)]
            for relation_tail in relation_tails:
                if relation_tail[0] in relation2id and relation_tail[1] in entity2id:
                    kg_idx[head].append((relation2id[relation_tail[0]], entity2id[relation_tail[1]]))

    return relation2id, kg_idx

def main(dataset: str = None):
    all_entity = set()

    file_list = [
        os.path.join(DIR, f'{dataset}/test_data_dbpedia.jsonl'),
        os.path.join(DIR, f'{dataset}/valid_data_dbpedia.jsonl'),
        os.path.join(DIR, f'{dataset}/train_data_dbpedia.jsonl'),
    ]
    for file in file_list:
        all_entity |= get_item_set(file)

    print(f'# all entity: {len(all_entity)}')

    with open(os.path.join(DIR, f'./kg.pkl'), 'rb') as f:
        kg = pkl.load(f)
    subkg = extract_subkg(kg, all_entity, 2)
    #print(subkg)
    relation2id, subkg = kg2id(subkg, dataset, all_entity)

    with open(os.path.join(DIR, f'{dataset}/dbpedia_subkg.json'), 'w', encoding='utf-8') as f:
        json.dump(subkg, f, ensure_ascii=False)
    # with open(os.path.join(DIR, f'{dataset}/entity2id.json'), 'w', encoding='utf-8') as f:
    #     json.dump(entity2id, f, ensure_ascii=False)
    with open(os.path.join(DIR, f'{dataset}/relation2id.json'), 'w', encoding='utf-8') as f:
        json.dump(relation2id, f, ensure_ascii=False)


if __name__ == '__main__':
    CLI(main)

