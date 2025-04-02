import json
import random

def extract_by_ratio_jsonl(input_file, output_file, ratio=0.1):
    # 读取所有行到列表
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"跳过无效行: {line[:50]}... 错误: {e}")
    
    # 检查数据是否为空
    if not data:
        print("文件为空或无有效数据")
        return
    
    # 计算要提取的数量
    extract_num = int(len(data) * ratio)
    if extract_num == 0:
        extract_num = 1  # 至少提取1条
    
    # 随机选择子集
    extracted_data = random.sample(data, min(extract_num, len(data)))
    
    # 保存为JSON格式（单一数组）
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in extracted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"从 {len(data)} 条数据中提取了 {len(extracted_data)} 条到 {output_file}")

# 使用示例
if __name__ == "__main__":
    input_file = "data/train_prompt_category.json"  # 你的输入文件
    output_file = "data/train_prompt_category_01.json"    # 输出文件
    extract_by_ratio_jsonl(input_file, output_file, 0.1)  # 提取1%