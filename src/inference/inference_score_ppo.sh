#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate readability_summ

export TOKENIZERS_PARALLELISM=false  # 添加这一行

VAL_FILE='../data/test_prompt_score_20.json'
MODEL_PATH='../train/rl/trlx/checkpoint-diverse/best_checkpoint/hf_model'

# 任务 1：Flesch Kincaid 得分 90
OUTPUT_DIR='outputs_ppo/1/'
CUDA_VISIBLE_DEVICES=0 python -u run_summarization_ppo.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_noprompt \
 --summary_column summary \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "Write highlights for this article with a flesch kincaid score of 90:\n\n" \
 --num_beams 4 \
 --overwrite_cache \

P1=$!

# 任务 2：Flesch Kincaid 得分 70
OUTPUT_DIR='outputs_ppo/2/'
CUDA_VISIBLE_DEVICES=1 python -u run_summarization_ppo.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_noprompt \
 --summary_column summary \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "Write highlights for this article with a flesch kincaid score of 70:\n\n" \
 --num_beams 4 \
 --overwrite_cache \

P2=$!

wait $P1 $P2

# 任务 3：Flesch Kincaid 得分 50
OUTPUT_DIR='outputs_ppo/3/'
CUDA_VISIBLE_DEVICES=0 python -u run_summarization_ppo.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_noprompt \
 --summary_column summary \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "Write highlights for this article with a flesch kincaid score of 50:\n\n" \
 --num_beams 4 \
 --overwrite_cache \

P3=$!

# 任务 4：Flesch Kincaid 得分 30
OUTPUT_DIR='outputs_ppo/4/'
CUDA_VISIBLE_DEVICES=1 python -u run_summarization_ppo.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_noprompt \
 --summary_column summary \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "Write highlights for this article with a flesch kincaid score of 30:\n\n" \
 --num_beams 4 \
 --overwrite_cache \

P4=$!

wait $P1 $P2 $P3 $P4

conda deactivate