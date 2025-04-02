from typing import List
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from readability import Readability
import numpy as np
import sys
eps = sys.float_info.epsilon
import math

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import textstat

model_dir = 'pretrained_models'  # select the checkpoint from the prompt-based methods

config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10,
        total_steps=100000,
        batch_size=2,
        # batch_size_eval=2,
        checkpoint_interval=10000,
        eval_interval=500,
        save_optimizer=False,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir='checkpoint-diverse',
        save_best=True
    ),
    model=ModelConfig(
        model_path=model_dir,
        model_arch_type="seq2seq",
        num_layers_unfrozen=-1,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=model_dir,
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 10000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=512,
        chunk_size=4,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.0,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 256,
        },
        gen_experience_kwargs={
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        },
    ),
)

def get_flesch(text):
    score = textstat.flesch_reading_ease(text)
    return score

import random

def change_scores(input_data):
    new_data = []
    for text in input_data:
        score_sum = random.choice([10, 15, 25, 30, 33, 35, 37, 40, 45, 48, 50, 52, 60, 64, 68, 70, 71, 75, 83, 84, 88, 89, 90, 92, 93, 94, 95])
        new_text = "Write highlights for this article with a flesch kincaid score of " + str(score_sum) + ":\n\n" + text
        new_data.append(new_text)
    return new_data

sigma = 10
def calc_nd(value, mean):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (value - mean) ** 2 / (2 * sigma ** 2)) / 0.039894228040143274


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
model_name = "roberta-large"
# model_name = "bart-large"
# device = "cuda:" + str(os.environ.get('LOCAL_RANK',0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_layers = 17
cache_dir = "roberta"
model = AutoModel.from_pretrained(cache_dir)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(cache_dir)
model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])

def encode_text(input_str):
    """
    Helper function to encode any string to tensor using the tokenizer
    """
    # inputs = tokenizer(input_str, padding=True, truncation=True, return_tensors="pt")
    inputs = tokenizer(input_str, padding='max_length', truncation=True, max_length=512, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # idf
    idf = torch.clone(inputs["attention_mask"]).float()
    idf[idf == tokenizer.sep_token_id] = 0
    idf[idf == tokenizer.cls_token_id] = 0
    idf.div_(idf.sum(dim=1, keepdim=True))

    return F.normalize(outputs[0], dim=-1), inputs["attention_mask"], idf

def compute_bertscore(doc_embedding, doc_masks, doc_idf, summ_embedding, summ_masks, summ_idf):
    """
    Helper function that is modified from the official code (greedy_cos_idf() method) https://github.com/Tiiiger/bert_score/blob/dbcf6db37e8bd6ff68446f06b0ba5d0763b62d20/bert_score/utils.py#L469
    """

    batch_size = doc_embedding.size(0)
    sim = torch.bmm(summ_embedding, doc_embedding.transpose(1, 2))
    masks = torch.bmm(summ_masks.unsqueeze(2).float(), doc_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    precision = sim.max(dim=2)[0]
    precision_scale = summ_idf.to(precision.device)
    P = (precision * precision_scale).sum(dim=1)

    summ_zero_mask = summ_masks.sum(dim=1).eq(2)
    if torch.any(summ_zero_mask):
        P = P.masked_fill(summ_zero_mask, 0.0)

    doc_zero_mask = doc_masks.sum(dim=1).eq(2)
    if torch.any(doc_zero_mask):
        P = P.masked_fill(doc_zero_mask, 0.0)

    return P


if __name__ == "__main__":

    # def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], tokenizer=None):

    #     flesch_scores = []
    #     original_scores = []
    #     summaries = []
    #     docs = []
    #     for (generated_summary, input_doc) in zip(outputs, prompts):
    #         score_sum = int(input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][:2].replace(":", ""))
    #         original_scores.append(score_sum)
    #         doc = input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][2:]
    #         docs.append(doc)
    #         summaries.append(generated_summary.strip())

    #         try:
    #             flesch_scores.append(get_flesch(generated_summary.strip()))
    #         except:
    #             flesch_scores.append(0)

    #     all_bertscore_scores = []
    #     for doc, summary in zip(docs, summaries):

    #         bertscore_input_embedding, bertscore_input_attention_mask, bertscore_input_idf = encode_text([doc])
    #         bertscore_output_embedding, bertscore_output_attention_mask, bertscore_output_idf = encode_text([summary])

    #         bertscore_scores = compute_bertscore(
    #             bertscore_input_embedding,
    #             bertscore_input_attention_mask,
    #             bertscore_input_idf,
    #             bertscore_output_embedding,
    #             bertscore_output_attention_mask,
    #             bertscore_output_idf,
    #         )
    #         bertscore_scores = bertscore_scores.tolist()
    #         all_bertscore_scores.extend(bertscore_scores)

    #     assert len(original_scores) == len(flesch_scores) == len(all_bertscore_scores)

    #     flesch_scores = [calc_nd(fs, o_fs) for fs, o_fs in zip(flesch_scores, original_scores)]

    #     readability_weight = 0.5
    #     flesch_scores = torch.tensor(flesch_scores)
    #     all_bertscore_scores = torch.tensor(all_bertscore_scores)
    #     flesch_scores = readability_weight * flesch_scores + (1 - readability_weight) * all_bertscore_scores
    #     flesch_scores = flesch_scores.tolist()

    #     return flesch_scores

    # def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], tokenizer=None, iter_count=0):
    #     flesch_scores = []
    #     original_scores = []
    #     summaries = []
    #     docs = []
    #     for (generated_summary, input_doc) in zip(outputs, prompts):
    #         # 解析提示
    #         try:
    #             score_sum = int(input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][:2].replace(":", ""))
    #         except (IndexError, ValueError):
    #             score_sum = 30  # 默认中等可读性
    #         original_scores.append(score_sum)
    #         doc = input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][2:]
    #         docs.append(doc)
    #         summaries.append(generated_summary.strip())

    #         # 计算 Flesch-Kincaid 分数
    #         try:
    #             fs = get_flesch(generated_summary.strip())
    #             flesch_scores.append(fs)
    #         except Exception as e:
    #             print(f"Flesch score error for summary '{generated_summary}': {e}")
    #             flesch_scores.append(30)

    #     # 计算 BERTScore
    #     all_bertscore_scores = []
    #     for doc, summary in zip(docs, summaries):
    #         bertscore_input_embedding, bertscore_input_attention_mask, bertscore_input_idf = encode_text([doc])
    #         bertscore_output_embedding, bertscore_output_attention_mask, bertscore_output_idf = encode_text([summary])

    #         bertscore_scores = compute_bertscore(
    #             bertscore_input_embedding,
    #             bertscore_input_attention_mask,
    #             bertscore_input_idf,
    #             bertscore_output_embedding,
    #             bertscore_output_attention_mask,
    #             bertscore_output_idf,
    #         )
    #         bertscore_scores = [max(bs, 0.5) for bs in bertscore_scores.tolist()]  # 确保最低分数
    #         all_bertscore_scores.extend(bertscore_scores)

    #     assert len(original_scores) == len(flesch_scores) == len(all_bertscore_scores)

    #     # 标准化 Flesch 分数
    #     flesch_scores = [calc_nd(fs, o_fs) for fs, o_fs in zip(flesch_scores, original_scores)]
        
    #     # 增加多样性奖励
    #     def diversity_bonus(summary):
    #         ngrams = [tuple(summary[i:i+2]) for i in range(len(summary)-1)]
    #         unique_ngrams = len(set(ngrams))
    #         total_ngrams = len(ngrams)
    #         diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    #         return diversity * 0.1

    #     # 动态探索奖励
    #     exploration_bonus = 0.1 if iter_count < 50000 else 0.05  # 早期鼓励探索

    #     # 平衡权重，增加探索
    #     readability_weight = 0.4  # 降低 Flesch 权重
    #     flesch_scores = torch.tensor(flesch_scores)
    #     all_bertscore_scores = torch.tensor(all_bertscore_scores)
    #     flesch_scores = (readability_weight * flesch_scores + 
    #                     (1 - readability_weight) * all_bertscore_scores + 
    #                     torch.tensor([diversity_bonus(s) for s in summaries]) + 
    #                     exploration_bonus)
        
    #     # 扩大奖励范围
    #     min_reward, max_reward = 0.0, 1.0
    #     current_min, current_max = flesch_scores.min().item(), flesch_scores.max().item()
    #     if current_max > current_min:
    #         flesch_scores = min_reward + (max_reward - min_reward) * (flesch_scores - current_min) / (current_max - current_min)
        
    #     flesch_scores = flesch_scores.tolist()
    #     return flesch_scores
    # def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], tokenizer=None, iter_count=0):
    #     flesch_scores = []
    #     original_scores = []
    #     summaries = []
    #     docs = []
        
    #     for (generated_summary, input_doc) in zip(outputs, prompts):
    #         try:
    #             score_sum = int(input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][:2].replace(":", ""))
    #         except (IndexError, ValueError):
    #             score_sum = 30
    #         original_scores.append(score_sum)
    #         doc = input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][2:]
    #         docs.append(doc)
    #         summaries.append(generated_summary.strip())
            
    #         try:
    #             fs = get_flesch(generated_summary.strip())
    #             flesch_scores.append(fs)
    #         except Exception as e:
    #             print(f"Flesch score error for summary '{generated_summary}': {e}")
    #             flesch_scores.append(30)

    #     # BERTScore 计算
    #     all_bertscore_scores = []
    #     for doc, summary in zip(docs, summaries):
    #         bertscore_input_embedding, bertscore_input_attention_mask, bertscore_input_idf = encode_text([doc])
    #         bertscore_output_embedding, bertscore_output_attention_mask, bertscore_output_idf = encode_text([summary])
    #         bertscore_scores = compute_bertscore(
    #             bertscore_input_embedding, bertscore_input_attention_mask, bertscore_input_idf,
    #             bertscore_output_embedding, bertscore_output_attention_mask, bertscore_output_idf,
    #         )
    #         bertscore_scores = [max(bs, 0.5) for bs in bertscore_scores.tolist()]
    #         all_bertscore_scores.extend(bertscore_scores)

    #     # 标准化 Flesch 分数
    #     flesch_scores = [calc_nd(fs, o_fs) for fs, o_fs in zip(flesch_scores, original_scores)]
        
    #     # 修改后的多样性奖励
    #     def diversity_bonus(summary, min_length=20):
    #         ngrams = [tuple(summary[i:i+2]) for i in range(len(summary)-1)]
    #         unique_ngrams = len(set(ngrams))
    #         total_ngrams = max(len(ngrams), min_length)
    #         diversity = unique_ngrams / total_ngrams
    #         return diversity * 0.1
        
    #     # 长度惩罚
    #     def length_penalty(summary, target_length=50):
    #         length = len(summary.split())
    #         length_score = -abs(length - target_length) / target_length
    #         return length_score * 0.2

    #     # 动态权重
    #     readability_weight = 0.3  # 降低 Flesch 权重
    #     exploration_bonus = 0.1 if iter_count < 50000 else 0.05

    #     # 综合奖励
    #     flesch_scores = torch.tensor(flesch_scores)
    #     all_bertscore_scores = torch.tensor(all_bertscore_scores)
    #     flesch_scores = (readability_weight * flesch_scores + 
    #                     (1 - readability_weight) * all_bertscore_scores + 
    #                     torch.tensor([diversity_bonus(s) for s in summaries]) + 
    #                     torch.tensor([length_penalty(s) for s in summaries]) + 
    #                     exploration_bonus)

    #     # 奖励标准化
    #     min_reward, max_reward = 0.0, 1.0
    #     current_min, current_max = flesch_scores.min().item(), flesch_scores.max().item()
    #     if current_max > current_min:
    #         flesch_scores = min_reward + (max_reward - min_reward) * (flesch_scores - current_min) / (current_max - current_min)

    #     return flesch_scores.tolist()
    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], tokenizer=None, iter_count=0):
        flesch_scores = []
        original_scores = []
        summaries = []
        docs = []
        ppl_scores = []  # 新增：存储PPL分数

        # 假设有一个语言模型可以计算PPL（需要传入模型或通过tokenizer访问）
        # 如果没有提供tokenizer，可以跳过PPL计算或使用默认值
        if tokenizer is None or not hasattr(tokenizer, 'model'):
            print("Warning: No tokenizer/model provided, skipping PPL calculation.")
            use_ppl = False
        else:
            use_ppl = True

        for (generated_summary, input_doc) in zip(outputs, prompts):
            # 解析提示
            try:
                score_sum = int(input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][:2].replace(":", ""))
            except (IndexError, ValueError):
                score_sum = 30  # 默认中等可读性
            original_scores.append(score_sum)
            doc = input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][2:]
            docs.append(doc)
            summaries.append(generated_summary.strip())

            # 计算 Flesch-Kincaid 分数
            try:
                fs = get_flesch(generated_summary.strip())
                flesch_scores.append(fs)
            except Exception as e:
                print(f"Flesch score error for summary '{generated_summary}': {e}")
                flesch_scores.append(30)

            # 计算 PPL（困惑度）
            if use_ppl:
                try:
                    # 使用tokenizer和模型计算PPL
                    inputs = tokenizer(generated_summary, return_tensors="pt")
                    with torch.no_grad():
                        outputs = tokenizer.model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss.item()
                        ppl = math.exp(loss)  # PPL = exp(loss)
                    ppl_scores.append(ppl)
                except Exception as e:
                    print(f"PPL calculation error for summary '{generated_summary}': {e}")
                    ppl_scores.append(50.0)  # 默认中等PPL值
            else:
                ppl_scores.append(50.0)  # 如果没有模型，默认值

        # 计算 BERTScore
        all_bertscore_scores = []
        for doc, summary in zip(docs, summaries):
            bertscore_input_embedding, bertscore_input_attention_mask, bertscore_input_idf = encode_text([doc])
            bertscore_output_embedding, bertscore_output_attention_mask, bertscore_output_idf = encode_text([summary])

            bertscore_scores = compute_bertscore(
                bertscore_input_embedding,
                bertscore_input_attention_mask,
                bertscore_input_idf,
                bertscore_output_embedding,
                bertscore_output_attention_mask,
                bertscore_output_idf,
            )
            bertscore_scores = [max(bs, 0.5) for bs in bertscore_scores.tolist()]  # 确保最低分数
            all_bertscore_scores.extend(bertscore_scores)

        assert len(original_scores) == len(flesch_scores) == len(all_bertscore_scores) == len(ppl_scores)

        # 标准化 Flesch 分数
        flesch_scores = [calc_nd(fs, o_fs) for fs, o_fs in zip(flesch_scores, original_scores)]
        
        # 标准化 PPL 分数（PPL越低越好，转换为奖励形式）
        if use_ppl:
            max_ppl = max(ppl_scores)  # 用于归一化
            ppl_scores = [1.0 - (ppl / max_ppl) if max_ppl > 0 else 1.0 for ppl in ppl_scores]  # 转换为0-1范围，越低越好
        else:
            ppl_scores = [0.5] * len(ppl_scores)  # 默认中等值

        # 增加多样性奖励
        def diversity_bonus(summary):
            ngrams = [tuple(summary[i:i+2]) for i in range(len(summary)-1)]
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
            return diversity * 0.1

        # 动态探索奖励
        exploration_bonus = 0.1 if iter_count < 50000 else 0.05  # 早期鼓励探索

        # 平衡权重，新增PPL权重
        readability_weight = 0.3  # 降低 Flesch 权重以容纳PPL
        bertscore_weight = 0.4   # BERTScore权重
        ppl_weight = 0.2         # 新增PPL权重
        diversity_weight = 0.1   # 多样性权重

        flesch_scores = torch.tensor(flesch_scores)
        all_bertscore_scores = torch.tensor(all_bertscore_scores)
        ppl_scores = torch.tensor(ppl_scores)

        # 综合奖励计算
        final_scores = (readability_weight * flesch_scores + 
                        bertscore_weight * all_bertscore_scores + 
                        ppl_weight * ppl_scores +
                        diversity_weight * torch.tensor([diversity_bonus(s) for s in summaries]) + 
                        exploration_bonus)
        
        # 扩大奖励范围
        min_reward, max_reward = 0.0, 1.0
        current_min, current_max = final_scores.min().item(), final_scores.max().item()
        if current_max > current_min:
            final_scores = min_reward + (max_reward - min_reward) * (final_scores - current_min) / (current_max - current_min)
        
        final_scores = final_scores.tolist()
        return final_scores


    train_file = '../../data/train_prompt_score.json'
    validation_file = '../../data/validation_prompt_score.json'
    data_files = {"train": train_file, "validation": validation_file}
    dataset = load_dataset("json", data_files=data_files)
    dataset['train'] = dataset['train'].shuffle(seed=42)
    dataset['validation'] = dataset['validation'].shuffle(seed=42)

    validation_examples = 2000
    val_prompts = [prompt for prompt in dataset['validation']["input_noprompt"][0:validation_examples]]
    print('\ntest 0\n', val_prompts[0])
    val_summaries = dataset['validation']["summary"][0:validation_examples]
    val_prompts = change_scores(val_prompts)
    assert len(val_prompts) == len(val_summaries)
    print('\ntest after 0 \n', val_prompts[0])

    prompts = dataset['train']["input_noprompt"]
    summaries = dataset['train']["summary"]
    prompts = [prompt for prompt in prompts]
    prompts = change_scores(prompts)
    assert len(prompts) == len(summaries)

    # make dictionary of prompts and labels to use for reward function
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    prompt_label = {}
    max_length = config.train.seq_length

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_prompts,
        config=config,
    )
