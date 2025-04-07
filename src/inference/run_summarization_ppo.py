#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk
import numpy as np
from datasets import load_dataset
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.utils import check_min_version, is_offline_mode
import textstat
from filelock import FileLock
from tqdm import tqdm  # 添加 tqdm 导入

# 导入 TRLX 相关模块
try:
    from trlx.models.modeling_ppo import AutoModelForSeq2SeqLMWithHydraValueHead
except ImportError:
    raise ImportError("TRLX is not installed. Please install it with `pip install trlx` to use PPO inference.")

logger = logging.getLogger(__name__)

check_min_version("4.28.0")

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

@dataclass
class ModelArguments:
    ppo_checkpoint: str = field(
        metadata={"help": "Path to the PPO-trained checkpoint (e.g., TRLX checkpoint directory)"}
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"})
    use_auth_token: bool = field(default=False, metadata={"help": "Will use the token generated when running `huggingface-cli login` (necessary for private models)."})

@dataclass
class DataArguments:
    test_file: str = field(
        metadata={"help": "The input test data file (a jsonlines or csv file) for inference."}
    )
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    text_column: Optional[str] = field(default=None, metadata={"help": "The name of the column in the dataset containing the input texts."})
    summary_column: Optional[str] = field(default=None, metadata={"help": "The name of the column in the dataset containing the reference summaries."})
    max_source_length: Optional[int] = field(default=1024, metadata={"help": "The maximum total input sequence length after tokenization."})
    val_max_target_length: Optional[int] = field(default=128, metadata={"help": "The maximum total sequence length for generated text."})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "Truncate the number of prediction examples for debugging."})
    num_beams: Optional[int] = field(default=4, metadata={"help": "Number of beams to use for generation."})
    source_prefix: Optional[str] = field(default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for preprocessing."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached preprocessed datasets."})

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(pred_ids, label_ids, tokenizer):
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # ROUGE 指标
    metric_rouge = evaluate.load("rouge")
    rouge_result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {k: round(v * 100, 4) for k, v in rouge_result.items()}

    # 可读性指标
    flesch_reading_ease = [textstat.flesch_reading_ease(pred) for pred in decoded_preds]
    gunning_fog = [textstat.gunning_fog(pred) for pred in decoded_preds]
    coleman_liau = [textstat.coleman_liau_index(pred) for pred in decoded_preds]

    readability_result = {
        "flesch_reading_ease": round(np.mean(flesch_reading_ease), 4),
        "gunning_fog": round(np.mean(gunning_fog), 4),
        "coleman_liau": round(np.mean(coleman_liau), 4),
    }

    # 生成长度
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in pred_ids]
    result = {"gen_len": round(np.mean(prediction_lens), 4)}

    result.update(rouge_result)
    result.update(readability_result)

    return result

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.info(f"PPO inference parameters: {training_args}")

    # 加载 PPO 模型和 tokenizer
    model = AutoModelForSeq2SeqLMWithHydraValueHead.from_pretrained(
        model_args.ppo_checkpoint,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.ppo_checkpoint,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # 加载测试数据集
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {"test": data_args.test_file}
        extension = data_args.test_file.split(".")[-1]
        if extension not in ["csv", "json"]:
            raise ValueError("`test_file` should be a csv or a json file.")
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
        )

    if "test" not in raw_datasets:
        raise ValueError("Test dataset is required for PPO inference.")

    predict_dataset = raw_datasets["test"]
    if data_args.max_predict_samples is not None:
        max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))

    # 数据预处理
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    column_names = predict_dataset.column_names
    text_column = data_args.text_column if data_args.text_column else column_names[0]
    summary_column = data_args.summary_column if data_args.summary_column else (column_names[1] if len(column_names) > 1 else None)
    if text_column not in column_names:
        raise ValueError(f"`text_column` '{data_args.text_column}' not found in dataset columns: {', '.join(column_names)}")
    if summary_column is None or summary_column not in column_names:
        logger.warning(f"No valid `summary_column` specified or found. Metrics requiring references (e.g., ROUGE) will be skipped.")
        compute_metrics_flag = False
    else:
        compute_metrics_flag = True

    def preprocess_function(examples):
        inputs = [prefix + inp for inp in examples[text_column]]
        model_inputs = tokenizer(
            inputs, 
            max_length=data_args.max_source_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
            return_attention_mask=True,
        )
        if compute_metrics_flag:
            labels = tokenizer(
                text_target=examples[summary_column], 
                max_length=data_args.val_max_target_length, 
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["labels"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]]
        return model_inputs

    # 添加进度条到数据集映射
    predict_dataset = predict_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Tokenizing dataset",
    )

    # 进行推理并收集预测结果
    model.eval()
    model.to(training_args.device)
    predictions = []
    all_pred_ids = []
    all_label_ids = [] if compute_metrics_flag else None

    # 添加进度条到推理循环
    with tqdm(total=len(predict_dataset), desc="Generating predictions") as pbar:
        for batch in predict_dataset:
            inputs = {k: torch.tensor([v]).to(training_args.device) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=data_args.val_max_target_length,
                    num_beams=data_args.num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                )
                outputs = torch.nn.functional.pad(
                    outputs, 
                    (0, data_args.val_max_target_length - outputs.shape[1]), 
                    value=tokenizer.pad_token_id, 
                )
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions.extend([pred.strip() for pred in decoded_outputs])
            all_pred_ids.extend(outputs.cpu().numpy())
            if compute_metrics_flag:
                labels = torch.tensor(batch["labels"]).unsqueeze(0)
                all_label_ids.extend(labels.cpu().numpy())
            pbar.update(1)  # 更新进度条
    
    # 保存预测结果
    output_prediction_file = os.path.join(training_args.output_dir, "ppo_generated_predictions.txt")
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))
        writer.flush()
    logger.info(f"PPO predictions saved to {output_prediction_file}")

    if compute_metrics_flag:
        print(f"Number of predictions: {len(all_pred_ids)}, Number of references: {len(all_label_ids)}")
        assert len(all_pred_ids) == len(all_label_ids), "Mismatch between predictions and references!"

        # 计算指标（如果有参考答案）
        metrics = compute_metrics(np.array(all_pred_ids), np.array(all_label_ids), tokenizer)
        logger.info(f"PPO inference metrics: {metrics}")
        # 保存指标到文件
        metrics_file = os.path.join(training_args.output_dir, "ppo_metrics.json")
        with open(metrics_file, "w") as f:
            import json
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
    else:
        logger.info("No reference summaries provided; skipping metric computation.")

if __name__ == "__main__":
    main()