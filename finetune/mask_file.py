import argparse
import re
import random
import csv

def extract_key_value_pairs(text):
    # 使用正则表达式匹配键值对，假设键和值由': '分隔，且每个词对之间由', '分隔
    pairs = re.findall(r"\[.*?\]", text)
    return pairs


def obfuscate_key_value_pairs(pairs, num_to_obfuscate):
    mask_pairs=[]
    # 随机选择要掩盖的词对数量
    random_indices = random.sample(range(len(pairs)), num_to_obfuscate)

    # 掩盖选中的词对
    for index in random_indices:
        pair = pairs[index]
        # 替换为["掩盖词", "掩盖词"]
        #pairs[index] = ["", ""]
        mask_pairs.append(pairs[index])

    return mask_pairs


def main():
    parser = argparse.ArgumentParser('Text Matching task')
    parser.add_argument('--input_file', default='../resource/redial/res.txt', type=str,
                        help='Input file path')
    parser.add_argument('--output_file', default='dataset.tsv', type=str,
                        help='Output file path')

    args = parser.parse_args()
    intput_path = args.input_file
    output_path=args.output_file
    # 假设文档内容是一行文本
    with open(intput_path,'r',encoding='utf-8') as fr:
        text=fr.readlines()
        random.shuffle(text)

        train_ratio = 0.8
        dev_ratio = 0.1
        test_ratio = 0.1

        # 计算每个数据集的大小
        total_data_size = len(text)
        train_size = int(total_data_size * train_ratio)
        dev_size = int(total_data_size * dev_ratio)
        test_size = int(total_data_size * test_ratio)

        # 用于存储TSV文件的列名
        header = ["sentence1", "sentence2", "score","split"]

        # 打开TSV文件进行写入
        with open(output_path, 'w', encoding='utf-8') as fw:
            # 写入表头
            fw.write("\t".join(header) + "\n")

            # 遍历文本行
            for i, l in enumerate(text):
                # 提取词对
                pairs = extract_key_value_pairs(l)

                # 要掩盖的词对数量
                num_to_obfuscate = random.randint(1, 3)

                # 掩盖词对
                obfuscated_pairs = obfuscate_key_value_pairs(pairs, num_to_obfuscate)
                replaced_string = l
                for mask_pair in obfuscated_pairs:
                    replaced_string = replaced_string.replace(mask_pair, '')

                split_type = ""
                if i <= train_size:
                    split_type = "train"
                elif i>train_size and i<= train_size+dev_size :
                    split_type = "dev"
                else:
                    split_type = "test"
                # 创建响应字典
                response = {
                    "sentence1": l.rstrip('\n'),
                    "sentence2": replaced_string.rstrip('\n'),
                    "score": 1,
                    "split": split_type
                }

                # 将响应写入TSV文件
                fw.write("\t".join([str(response.get(key, "")) for key in header]) + "\n")

if __name__ == "__main__":
    main()
