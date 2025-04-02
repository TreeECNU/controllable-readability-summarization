import pandas as pd

# # 读取并合并训练数据
# train1 = pd.read_parquet('data/train-00000-of-00003.parquet')
# train2 = pd.read_parquet('data/train-00001-of-00003.parquet')
# train3 = pd.read_parquet('data/train-00002-of-00003.parquet')
# train = pd.concat([train1, train2, train3], ignore_index=True)

# # 将训练数据保存为 JSON Lines 文件
# train.to_json('data/train.jsonl', orient='records', lines=True, force_ascii=False)

# # 读取并保存验证数据
# valid = pd.read_parquet('data/validation-00000-of-00001.parquet')
# valid.to_json('data/validation.jsonl', orient='records', lines=True, force_ascii=False)

# # 读取并保存测试数据
# test = pd.read_parquet('data/test-00000-of-00001.parquet')
# test.to_json('data/test.jsonl', orient='records', lines=True, force_ascii=False)

# 预览json中的前5个
print(pd.read_json('data/train.jsonl', lines=True).head())

# 打印出第一行的内容
# print(pd.read_json('data/train_prompt_score.json', lines=True).head())

# 判断是否在使用GPU
import torch
# print(torch.cuda.is_available())