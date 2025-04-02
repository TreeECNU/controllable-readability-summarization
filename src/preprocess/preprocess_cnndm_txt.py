import json
from readability import Readability
import nltk

# 下载必要的 NLTK 资源
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# 解析一行文本，提取文章和摘要
def parse_line(line):
    parts = line.split('<summ-content>', 1)
    article = ''.join(parts[0].split('<s>')[1:]).replace('</s>', '').strip()
    summary = parts[1].strip() if len(parts) > 1 else ''
    return article, summary

# 计算可读性指标
def compute_metrics(text):
    metrics = {}
    r = Readability(text)
    # 计算所有需要的可读性指标
    metrics['flesch'] = round(r.flesch().score, 4)
    metrics['dale_chall'] = round(r.dale_chall().score, 4)
    metrics['coleman_liau'] = round(r.coleman_liau().score, 4)
    metrics['gunning_fog'] = round(r.gunning_fog().score, 4)
    return metrics

# 处理数据集中的某个部分（训练集、验证集或测试集）
def process_data(split):
    input_file = f'{split}.txt'
    output_file = f'{split}.jsonl'  # 使用.jsonl扩展名表示JSON Lines格式
    data = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            article, summary = parse_line(line)
            entry = {
                'id': str(idx),  # 使用行号作为ID
                'input': article,
                'input_metrics': compute_metrics(article),
                'summary': summary,
                'summary_metrics': compute_metrics(summary.replace("\n", " "))
            }
            data.append(entry)

    # 将处理后的数据保存到文件中
    save_file(data, output_file)

# 保存数据到文件
def save_file(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# 处理各个数据集部分
process_data('data/cnndm-pj/train')
process_data('data/cnndm-pj/validation')
process_data('data/cnndm-pj/test')