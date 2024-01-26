# import re
#
# import json
# from nltk import ngrams
# from nltk.translate.bleu_score import sentence_bleu
# from jsonargparse import CLI
# from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
#
# # year_pattern = re.compile(r'/(/d{4}/)')
# slot_pattern = re.compile(r'<movie>')
#
#
# class ConvEvaluator:
#     def __init__( tokenizer, log_file_path):
#         tokenizer = tokenizer
#
#         reset_metric()
#         if log_file_path:
#             log_file = open(log_file_path, 'w', buffering=1)
#             log_cnt = 0
#
#     def evaluate( preds, labels, log=False):
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
#         decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
#                          decoded_preds]
#         decoded_preds = [pred.strip() for pred in decoded_preds]
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
#         decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in
#                           decoded_labels]
#         decoded_labels = [label.strip() for label in decoded_labels]
#
#         if log and hasattr( 'log_file'):
#             for pred, label in zip(decoded_preds, decoded_labels):
#                 log_file.write(json.dumps({
#                     'pred': pred,
#                     'label': label
#                 }, ensure_ascii=False) + '/n')
#
#         collect_ngram(decoded_preds)
#         compute_item_ratio(decoded_preds)
#         compute_bleu(decoded_preds, decoded_labels)
#         sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])
#
#     def collect_ngram( strs):
#         for str in strs:
#             str = str.split()
#             for k in range(1, 5):
#                 dist_k = f'dist@{k}'
#                 for token in ngrams(str, k):
#                     metric[dist_k].add(token)
#
#     def compute_bleu( preds, labels):
#         for pred, label in zip(preds, labels):
#             pred, label = pred.split(), [label.split()]
#             for k in range(4):
#                 weights = [0] * 4
#                 weights[k] = 1
#                 metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)
#
#     def compute_item_ratio( strs):
#         for str in strs:
#             # items = re.findall(year_pattern, str)
#             # metric['item_ratio'] += len(items)
#             items = re.findall(slot_pattern, str)
#             metric['item_ratio'] += len(items)
#
#     def report(:
#         report = {}
#         for k, v in metric.items():
#             if sent_cnt == 0:
#                 report[k] = 0
#             else:
#                 if 'dist' in k:
#                     v = len(v)
#                 report[k] = v / sent_cnt
#         report['sent_cnt'] = sent_cnt
#         return report
#
#     def reset_metric(:
#         metric = {
#             'bleu@1': 0,
#             'bleu@2': 0,
#             'bleu@3': 0,
#             'bleu@4': 0,
#             'dist@1': set(),
#             'dist@2': set(),
#             'dist@3': set(),
#             'dist@4': set(),
#             'item_ratio': 0,
#         }
#         sent_cnt = 0
#
# def main(from_json: str = None):
#     tokenizer = LlamaTokenizer.from_pretrained(
#                 '/home/dell/qsj/LLMS/Llama2-Chinese-7b-Chat',
#                 trust_remote_code=True,
#                 device_map="cpu"
#             )
#     resp = [json.loads(l)["resp"] for l in open(from_json)]
#     evaluator = ConvEvaluator(tokenizer=tokenizer)
#     for i,l in enumerate(resp):
#         gen_resp_ids = tokenizer(l, return_tensors="pt")
#         evaluator.evaluate(gen_resp_ids, l)
#
#     report = evaluator.report()
#     print(report)
#
# if __name__ == '__main__':
#     CLI(main)

# import json
# from nltk.translate.bleu_score import sentence_bleu
#
#
# def individual_bleu(reference, candidate):
#     bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#     bleu_2_gram = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
#     bleu_3_gram = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
#     bleu_4_gram = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
#
#     # print('bleu 1-gram: %f' % bleu_1_gram)
#     # print('bleu 2-gram: %f' % bleu_2_gram)
#     # print('bleu 3-gram: %f' % bleu_3_gram)
#     # print('bleu 4-gram: %f' % bleu_4_gram)
#
#     return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram
#
#
# from nltk.translate.bleu_score import sentence_bleu
# def cumulative_bleu(reference, candidate):
#
#     bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#     bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
#     bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
#     bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
#
#     # print('bleu 1-gram: %f' % bleu_1_gram)
#     # print('bleu 2-gram: %f' % bleu_2_gram)
#     # print('bleu 3-gram: %f' % bleu_3_gram)
#     # print('bleu 4-gram: %f' % bleu_4_gram)
#
#     return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram
#
# eval_file = "E:/pythonFiles/files/LLM+kg/src/gpt-3.5/c0/db/inspired/extracted.jsonl"
# gt_file = "E:/pythonFiles/files/LLM+kg/data/inspired/test.jsonl"
# sents = []
# preds = [json.loads(l)["resp"] for l in open(eval_file)]
# gts = [json.loads(l)["resp"] for l in open(gt_file)]
# # # 生成文本
# # generated_text = "This is some generated text."
# #
# # # 参考文本列表
# # reference_texts = ["This is a reference text.", "This is another reference text."]
#
# for p,r in zip(preds, gts):
#     # 计算 Bleu 指标
#     i_bleu = individual_bleu(r, p)
#     c_bleu = cumulative_bleu(r, p)
#
#     # 打印结果
#     print("The Bleu score is:", i_bleu)
#     print("The Bleu score is:", c_bleu)

# !/usr/bin/env python
# -*- coding: utf-8 -*-


# import sys
# import math
# import json
# from collections import Counter
#
# if len(sys.argv) < 2:
#     print("Usage: " + sys.argv[0] + " eval_file")
#     print("eval file format: pred_response /t gold_response")
#     exit()
#
#
# def get_dict(tokens, ngram, gdict=None):
#     """
#     get_dict
#     统计n-gram频率并用dict存储
#     """
#     token_dict = {}
#     if gdict is not None:
#         token_dict = gdict
#     tlen = len(tokens)
#     for i in range(0, tlen - ngram + 1):
#         ngram_token = "".join(tokens[i:(i + ngram)])
#         if token_dict.get(ngram_token) is not None:
#             token_dict[ngram_token] += 1
#         else:
#             token_dict[ngram_token] = 1
#     return token_dict
#
#
# def count(pred_tokens, gold_tokens, ngram, result):
#     """
#     计算BLEU中pn
#     """
#     cover_count, total_count = result
#     pred_dict = get_dict(pred_tokens, ngram)
#     gold_dict = get_dict(gold_tokens, ngram)
#     cur_cover_count = 0
#     cur_total_count = 0
#     for token, freq in pred_dict.items():
#         if gold_dict.get(token) is not None:
#             gold_freq = gold_dict[token]
#             cur_cover_count += min(freq, gold_freq)
#         cur_total_count += freq
#     result[0] += cur_cover_count
#     result[1] += cur_total_count
#
#
# def calc_bp(pair_list):
#     """
#     calc_bp
#     """
#     c_count = 0.0
#     r_count = 0.0
#     for pair in pair_list:
#         pred_tokens, gold_tokens = pair
#         c_count += len(pred_tokens)
#         r_count += len(gold_tokens)
#     bp = 1
#     if c_count < r_count:
#         bp = math.exp(1 - r_count / c_count)
#     return bp
#
#
# def calc_cover_rate(pair_list, ngram):
#     """
#     calc_cover_rate
#     """
#     result = [0.0, 0.0]  # [cover_count, total_count]
#     for pair in pair_list:
#         pred_tokens, gold_tokens = pair
#         count(pred_tokens, gold_tokens, ngram, result)
#     cover_rate = result[0] / result[1]
#     return cover_rate
#
#
# def calc_bleu(pair_list):
#     """
#     calc_bleu
#     """
#     bp = calc_bp(pair_list)
#     cover_rate1 = calc_cover_rate(pair_list, 1)
#     cover_rate2 = calc_cover_rate(pair_list, 2)
#     cover_rate3 = calc_cover_rate(pair_list, 3)
#     bleu1 = 0
#     bleu2 = 0
#     bleu3 = 0
#     if cover_rate1 > 0:
#         bleu1 = bp * math.exp(math.log(cover_rate1))
#     if cover_rate2 > 0:
#         bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
#     if cover_rate3 > 0:
#         bleu3 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3)
#     return [bleu1, bleu2, bleu3]
#
#
# def calc_distinct_ngram(pair_list, ngram):
#     """
#     calc_distinct_ngram
#     """
#     ngram_total = 0.0
#     ngram_distinct_count = 0.0
#     pred_dict = {}
#     for predict_tokens, _ in pair_list:
#         get_dict(predict_tokens, ngram, pred_dict)
#     for key, freq in pred_dict.items():
#         ngram_total += freq
#         ngram_distinct_count += 1
#         # if freq == 1:
#         #    ngram_distinct_count += freq
#     return ngram_distinct_count / ngram_total
#
#
# def calc_distinct(pair_list):
#     """
#     calc_distinct
#     """
#     distinct1 = calc_distinct_ngram(pair_list, 1)
#     distinct2 = calc_distinct_ngram(pair_list, 2)
#     return [distinct1, distinct2]
#
#
# def calc_f1(data):
#     """
#     calc_f1
#     """
#     golden_char_total = 0.0
#     pred_char_total = 0.0
#     hit_char_total = 0.0
#     for response, golden_response in data:
#         # golden_response = "".join(golden_response).decode("utf8")
#         # response = "".join(response).decode("utf8")
#         golden_response = "".join(golden_response)
#         response = "".join(response)
#         common = Counter(response) & Counter(golden_response)
#         hit_char_total += sum(common.values())
#         golden_char_total += len(golden_response)
#         pred_char_total += len(response)
#     p = hit_char_total / pred_char_total
#     r = hit_char_total / golden_char_total
#     f1 = 2 * p * r / (p + r)
#     return f1
#
#
# eval_file = "E:/pythonFiles/files/LLM+kg/src/gpt-3.5/c0/db/inspired/extracted.jsonl"
# gt_file = "E:/pythonFiles/files/LLM+kg/data/inspired/test.jsonl"
# sents = []
# preds = [json.loads(l)["resp"] for l in open(eval_file)]
# gts = [json.loads(l)["resp"] for l in open(gt_file)]
# for p,r in zip(preds,gts):
#     tk = [p,r]
#     #print(tk)
#     if len(tk) < 2:
#         continue
#     pred_tokens = tk[0].strip().split(" ")
#     gold_tokens = tk[1].strip().split(" ")
#     sents.append([pred_tokens, gold_tokens])
# # calc f1
# f1 = calc_f1(sents)
# # calc bleu
# bleu1, bleu2, bleu3 = calc_bleu(sents)
# # calc distinct
# distinct1, distinct2 = calc_distinct(sents)
#
# output_str = "F1: %.2f%%/n" % (f1 * 100)
# output_str += "BLEU1: %.3f%%/n" % bleu1
# output_str += "BLEU2: %.3f%%/n" % bleu2
# output_str += "DISTINCT1: %.3f%%/n" % distinct1
# output_str += "DISTINCT2: %.3f%%/n" % distinct2
# sys.stdout.write(output_str)

from base import Metrics, BaseEvaluator
from gen import F1Metric, BleuMetric, AverageMetric
from nltk import ngrams
import json
from collections import defaultdict

eval_file = "E:/pythonFiles/files/LLM+kg/src/glm/c0/conv/raw/redial/extracted.jsonl"
gt_file = "E:/pythonFiles/files/LLM+kg/data/redial/test.jsonl"
sents = []
preds = [json.loads(l)["preference"] for l in open(eval_file)]
gts = [json.loads(l)["resp"] for l in open(gt_file)]

# dist_set = defaultdict(set)
# dist_cnt = 0
# gen_metrics = Metrics()
# optim_metrics = Metrics()
#
#
# def gen_evaluate(hyp, refs):
#     if hyp:
#         gen_metrics.add("f1", F1Metric.compute(hyp, refs))
#
#         for k in range(1, 5):
#             gen_metrics.add(f"bleu@{k}", BleuMetric.compute(hyp, refs, k))
#             # split sentence to tokens here
#             hyp_token = hyp.split()
#             for token in ngrams(hyp_token, k):
#                 dist_set[f"dist@{k}"].add(token)
#         dist_cnt += 1
#
# def conv_evaluate(prediction, response):
#     prediction = prediction.tolist()
#     response = response.tolist()
#     for p, r in zip(prediction, response):
#         p_str = p
#         r_str = r
#         gen_evaluate(p_str, [r_str])
#
#
# def report():
#     for k, v in dist_set.items():
#         gen_metrics.add(k, AverageMetric(len(v) / dist_cnt))
#     reports = [gen_metrics.report(), optim_metrics.report()]
#     print(reports)
#
#conv_evaluate(preds,gts)


class ConvEvaluator(BaseEvaluator):
    """The evaluator specially for conversational model

    Args:
        dist_set: the set to record dist n-gram
        dist_cnt: the count of dist n-gram evaluation
        gen_metrics: the metrics to evaluate conversational model, including bleu, dist, embedding metrics, f1
        optim_metrics: the metrics to optimize in training

    """

    def __init__(self, tensorboard=False):
        super(ConvEvaluator, self).__init__()
        self.dist_set = defaultdict(set)
        self.dist_cnt = 0
        self.gen_metrics = Metrics()
        self.optim_metrics = Metrics()

    def gen_evaluate(self, hyp, refs):

        if hyp:
            self.gen_metrics.add("f1", F1Metric.compute(hyp, refs))

            for k in range(1, 5):
                self.gen_metrics.add(f"bleu@{k}", BleuMetric.compute(hyp, refs, k))
                # split sentence to tokens here
                hyp_token = hyp[:len(refs[0])].split()
                for token in ngrams(hyp_token, k):
                    self.dist_set[f"dist@{k}"].add(token)
            self.dist_cnt += 1

            # hyp_emb = self._get_sent_embedding(hyp)
            # ref_embs = [self._get_sent_embedding(ref) for ref in refs]
            # self.gen_metrics.add('greedy', GreedyMatch.compute(hyp_emb, ref_embs))
            # self.gen_metrics.add('average', EmbeddingAverage.compute(hyp_emb, ref_embs))
            # self.gen_metrics.add('extreme', VectorExtrema.compute(hyp_emb, ref_embs))

    def report(self):
        for k, v in self.dist_set.items():
            self.gen_metrics.add(k, AverageMetric(len(v) / self.dist_cnt))
        reports = [self.gen_metrics.report(), self.optim_metrics.report()]
        print(reports)
        # if self.tensorboard and mode != 'test':
        #     for idx, task_report in enumerate(reports):
        #         for each_metric, value in task_report.items():
        #             self.writer.add_scalars(f'{self.reports_name[idx]}/{each_metric}', {mode: value.value()}, epoch)
        #
        # logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))

    def reset_metrics(self):
        self.gen_metrics.clear()
        self.dist_cnt = 0
        self.dist_set.clear()
        self.optim_metrics.clear()

conv_eva = ConvEvaluator()
def conv_evaluate(prediction, response):
    # prediction = prediction.tolist()
    # response = response.tolist()
    for p, r in zip(prediction, response):
        p_str = p
        r_str = r
        #print(p_str, r_str)
        if p_str!="":
            # print(p_str)
            # print(r_str)
            conv_eva.gen_evaluate(p_str, [r_str])

conv_evaluate(preds,gts)
conv_eva.report()