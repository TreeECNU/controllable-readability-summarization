import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)

# import os

# train_file = '../../data/train_prompt_score.json'
# validation_file = '../../data/validation_prompt_score.json'
# if not os.path.exists(train_file):
#     print(f"训练文件不存在: {train_file}")
# if not os.path.exists(validation_file):
#     print(f"验证文件不存在: {validation_file}")

