import json
import os
import uuid
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import Clock, infinite_dataloader
from trlx.utils.modeling import RunningMoments, gather_dict, logprobs_of_labels

logger = logging.get_logger(__name__)


@register_trainer
class AcceleratePPOTrainer(AccelerateRLTrainer):
    # 继承自 AccelerateRLTrainer，这是一个基于 accelerate 的强化学习基类。
    """PPO Accelerate Trainer"""

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """PPO Accelerate Trainer initialization

        Args:
            config: `TRLConfig`
            kwargs: Additional keyword arguments passed to `AccelerateRLTrainer`
        """
        super().__init__(config, **kwargs)

        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        
        # 创建 PPORolloutStorage 实例，用于存储 PPO 的 rollout 数据（提示、响应、日志概率、值、奖励等）。
        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        self.store = PPORolloutStorage(self.tokenizer.pad_token_id, self.tokenizer.padding_side)

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(self.config.train.batch_size, shuffle=True)

        # Prepare multi-GPU acceleration
        # 使用 accelerator 准备模型、优化器、调度器和数据加载器，以支持分布式训练。
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Set up a reference model when hydra heads are not used
        # 如果模型没有冻结头或不是 PEFT 类型，创建参考模型用于 KL 散度计算。
        if not hasattr(self.model, "frozen_head") and not self.model.peft_type:
            self.ref_model = self.get_arch(self.config)
            self.ref_model.to(self.accelerator.device)
            self.ref_model.eval()

        # Set up the KL controller
        # This helps prevent large divergences in the controller (policy)
        # 根据配置选择自适应或固定 KL 控制器，控制策略与参考模型的散度。
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(config.method.init_kl_coef, config.method.target, config.method.horizon)
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        # 配置生成参数（如采样、缓存、终止 token），支持模型生成。
        generate_kwargs = dict(
            do_sample=True,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            synced_gpus=os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3",
        )
        self.generate_kwargs = {**generate_kwargs, **config.method.gen_kwargs}

        if config.method.gen_experience_kwargs is not None:
            self.generate_experience_kwargs = {**generate_kwargs, **config.method.gen_experience_kwargs}
        else:
            self.generate_experience_kwargs = None

        # Setup stats tracker
        # 使用 RunningMoments 跟踪奖励的均值和标准差。
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_arch(self, config: TRLConfig):
        """Returns a specific wrapper given a model's architecture"""
        # 根据模型架构（因果语言模型或Seq2Seq模型）返回适当的模型类，并从预训练路径或配置加载。
        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config

        return from_fn(
            config.model.model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
            num_value_layers_unfrozen=config.method.num_value_layers_unfrozen,
            peft_config=self.config.model.peft_config,
            **self.config.model.model_extra_configs,
        )

    def loss(self, batch: PPORLBatch) -> Tuple[float, Dict[str, Any]]:
        """Computes loss on a batch of data and returns statistics
            计算一个批次的损失并返回统计信息。
        Args:
            batch: `PPORLBatch` Previous batch of episodes

        Returns:
            loss: `Float` Loss value
            stats: `Dict[str, Any]` PPO Statistics values
        """

        # Move `batch` data to `accelerator` device
        # 数据移动到GPU上
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)
        response_length = old_rewards.shape[1]

        # 计算advantages（当前动作相对于基线的优劣）和returns（每一步的累计奖励）
        advantages, returns = self.config.method.get_advantages_and_returns(old_values, old_rewards, response_length)

        if self.config.model.model_arch_type == "seq2seq":
            input_ids = query_tensors
            decoder_input_ids = response_tensors
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            decoder_attention_mask = (
                decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            )
            decoder_attention_mask[:, 0] = 1

            # Forward pass
            # 前向传播，获取outputs
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            # 准备PPO损失的输入，包括logits、values_pred、logprobs、mask
            logits = outputs.logits
            values_pred = outputs.value
            logprobs = logprobs_of_labels(logits[:, :-1, :], decoder_input_ids[:, 1:])
            mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            start = 0
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                mask[:, start + 1 : end + 1],
            )
        else:
            tokens = torch.cat((query_tensors, response_tensors), dim=1)
            attention_mask = tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = self.model(tokens, attention_mask, return_dict=True, position_ids=position_ids)
            logits = outputs.logits
            values_pred = outputs.value
            values_pred = values_pred[:, :-1]
            logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

            start = query_tensors.shape[1] - 1
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start + 1 : end + 1],
            )
        # 计算PPO损失
        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )

        return loss, stats

    def setup_rollout_logging(self, config):
        """Make rollout logging directory to log rollouts to"""
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(config.train.rollout_logging_dir, self.run_id)
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        """Clears the rollout store and creates `num_rollouts` new samples"""
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

    def post_backward_callback(self):
        self.kl_ctl.update(self.mean_kl, n_steps=self.config.train.batch_size)

    def create_train_dataloader(self):
        return self.store.create_loader(self.config.train.batch_size, shuffle=True)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.method.chunk_size)
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)

        self.make_experience(self.config.method.num_rollouts)

        self.train_dataloader = self.create_train_dataloader()

        self.n_inner_epochs = self.config.method.ppo_epochs
        self.total_steps = self.config.train.epochs * self.n_inner_epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=True)
        prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = infinite_dataloader(prompt_dataloader)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """
        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates for all batches & epochs
        """
        # make_experience 方法的主要目的是：
        # 从提示数据集（prompt_iterator）中获取批次数据。
        # 使用当前模型生成响应（samples）。
        # 计算奖励（rewards）、对数概率（logprobs）、值函数估计（values）等信息。
        # 计算当前模型与参考模型之间的 KL 散度（作为惩罚项）。
        # 将生成的经验数据封装为 PPORLElement 对象并存储到训练器的缓冲区中（self.store）。
        
        # 初始化和日志设置
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        ppo_rl_elements = []
        accumulated_stats = []

        # 生成经验数据
        while len(ppo_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            # 从 self.prompt_iterator 获取一个提示批次（batch）
            # 包含 input_ids 和 attention_mask 等。
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            # 生成样本
            samples = self.generate(batch["input_ids"], batch["attention_mask"])
            stats["time/rollout_generate"] = time() - rollout_generate_time

            # 数据准备和分布式同步
            # input_ids = (batch_size, sequence_length)
            prompt_tensors = batch.input_ids
            device = samples.device
            
            # prompt_tensors.shape[1] -> sequence_length
            # len(prompt_tensors) -> batch_size
            # prompt_sizes = (batch_size,)
            # 如果 prompt_tensors 是 (4, 10)，则 prompt_sizes 是 (4,)，值为 [10, 10, 10, 10]。
            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            # 填充samples
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            # 填充prompts
            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            # 将samples和prompts收集到主进程中
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            # 从batch中提取出除input_ids和attention_mask之外的其他数据，并进行分布式同步
            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask"})
            
            # 主流程：解码生成的样本和计算reward
            # 解码和reward计算只在主进程执行
            if self.accelerator.is_main_process:
                # 将张量形式的提示和样本解码为字符串
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )

                rollout_score_time = time()
                # reward_fn should return list of rewards at each token per sample
                # NOTE: all_scores[0][i] is the reward due to token (action) i in prompt + response (b/c of how kl is computed)
                # 调用奖励函数计算所有samples的reward
                all_scores = self.reward_fn(
                    samples=all_str_samples,
                    prompts=all_str_prompts,
                    outputs=all_str_outputs,
                    tokenizer=self.tokenizer,
                    **metadata,
                )
                # 将奖励列表转换为张量列表
                all_scores = [
                    torch.tensor(score, dtype=torch.float, device=device).view(
                        -1,
                    )
                    for score in all_scores
                ]
                # Pad 0 reward on the ends
                # 将不同长度的奖励张量填充到相同长度
                all_scores = pad_sequence(all_scores, batch_first=True, padding_value=-np.inf)
                # 计算填充后奖励张量的最大长度
                max_len = torch.tensor(all_scores.shape[1], dtype=torch.long, device=device)
                stats["time/rollout_score"] = time() - rollout_score_time
                # 将填充后的奖励张量按进程数分割
                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1, max_len).unbind())
            else:
                all_scores = None
                max_len = torch.tensor(0, dtype=torch.long, device=device)

            # 分布式广播和散布
            if torch.distributed.is_initialized():
                # 将主进程中的max_len广播到所有进程
                torch.distributed.broadcast(max_len, 0)
                # 为当前进程分配一个空的奖励张量scores
                scores = torch.empty((len(samples), max_len), device=device)
                # 将主进程的all_scores分散到每个进程的scores中
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()
            # 创建布尔掩码，标记奖励张量中的有效值
            scores_mask = scores != -np.inf

            # 本地解码和输出处理
            # 在当前进程中，将本地提示和样本解码为字符串
            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)

            # Pad the sample outputs
            # 将outputs中的字符串重新分词为token ID，转换为张量形式，便于后续填充和计算
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                # 在每个输出的开头添加<pad>标记
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]
            # 将token ID列表转换为张量列表，并计算最大长度
            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            # 将所有输出张量填充到相同长度
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            # 将填充后的输出张量堆叠为一个二维张量
            sample_outputs = torch.vstack(outputs).to(device)

            # 奖励裁剪和标准化
            # 将奖励裁剪到上下范围中，防止奖励过大或过小，稳定训练
            if self.config.method.cliprange_reward:
                scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = (scores * scores_mask).sum(dim=1).mean(), (scores * scores_mask).sum(
                    dim=1
                ).std()
            # 更新运行时的奖励统计
            all_scores_mean, all_scores_std = self.running_moments.update(torch.sum(scores * scores_mask, dim=1))
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            # 计算logprobs和values
            # Precompute logprobs, values
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                # 创建解码器的注意力掩码，标记非填充位置
                decoder_attention_mask = sample_outputs.not_equal(self.tokenizer.pad_token_id)
                decoder_attention_mask[:, 0] = 1
                with torch.no_grad():
                    # 使用模型进行前向传播，获取logits和values
                    outputs = self.model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    # 检查模型是否具有冻结头或使用PEFT（参数高效微调）
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        # 如果有冻结头或者PEFT，使用模型的forward_hydra方法计算ref_logits
                        ref_logits = self.model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        # 否则，使用独立的参考模型计算ref_logits
                        ref_logits = self.ref_model(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
            else:
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                with torch.no_grad():
                    logits, *_, values = self.model(
                        all_tokens, attention_mask=attention_mask, position_ids=position_ids
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                        ref_logits = ref_logits.to(device)
            
            # 计算KL散度和最终奖励
            if self.config.model.model_arch_type == "seq2seq":
                # 生成当前logits的对数概率
                # logits[:, :-1, :]：去除最后一个时间步的预测
                #（形状 (batch_size, seq_len-1, vocab_size)），因为它预测的是下一个 token。
                # sample_outputs[:, 1:]：实际采样 token，去除第一个（通常是 <pad>），
                # 形状 (batch_size, seq_len-1)。
                # logprobs_of_labels：trlx.utils 中的函数，
                # 从 logits 中提取对应标签（sample_outputs）的对数概率。
                # 输出：logprobs 是 (batch_size, seq_len-1) 的张量，
                # 表示每个 token 的对数概率。
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                # 计算当前参考logits的对数概率
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
            else:
                # NOTE: logprob[i] is (log)prob at which all_token[i+1] was sampled
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])
            # 获取当前batch的sample数量
            n_samples: int = samples.shape[0]

            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                # 为seq2seq模型生成注意力掩码
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = prompt_tensors.shape[1] - 1
            # 计算对数概率比率，并且应用注意力掩码
            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            # 计算KL散度
            kl = log_ratio.exp() - 1 - log_ratio
            # 计算平均逐 token KL 散度
            mean_kl_per_token = kl.mean()
            # 计算每个样本的平均 KL 散度
            mean_kl = kl.sum(1).mean()

            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.cpu()[:, :-1]

            # Get the logprobs and values, for tokens that are not padding,
            # from the end of the prompt up to the <eos> token, while also including the latter
            # (these are taken from the student model and not the reference model)
            # 构建PPO元素
            # 计算每个样本的结束位置（包括<eos>），确定有效序列范围
            ends = start + attention_mask[:, start:].sum(1) + 1
            # 提取每个样本的有效values
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            # 提取每个样本的有效 logprobs
            all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]
            # 生成奖励的KL惩罚部分
            kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
            kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]

            rollout_count = 0
            # 构建PPO元素
            for sample_idx in range(n_samples):
                # 将KL惩罚作为奖励的基准
                rewards = kl_penalty[sample_idx]
                # Then add in rewards
                # 将外部奖励（scores）添加到rewards中
                if scores.shape[1] == 1:
                    # NOTE: Final reward given at EOS token following HHH practice
                    # 将单个奖励加到序列的最后一个token上
                    rewards[-1] += scores[sample_idx][0].cpu()
                else:
                    # 处理逐token奖励的情况
                    # 提取当前样本的奖励张量
                    score = scores[sample_idx]
                    # 计算当前样本有效奖励的长度
                    score_right_padding = torch.sum(scores_mask[sample_idx])
                    # 切片有效奖励并移动到CPU中
                    score = score[:score_right_padding].cpu()
                    # 创建与rewards形状相同的零张量
                    p_score = torch.zeros_like(rewards)
                    # 将有效奖励添加到p_score的前部
                    p_score[: score.shape[0]] += score
                    # 将逐token奖励累加到rewards上
                    rewards += p_score

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )

                rollout_count += 1

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_kl, torch.distributed.ReduceOp.AVG)

            # 统计和储存
            stats["time/rollout_time"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        """
        Args:
            directory (str, *optional*): The directory to save the trainer files to.
                NOTE: If not specified, the model will be saved to a directory named `hf_model` in the
                checkpoint directory as specified by the Trainer's config.
            **kwargs: Additional keyword arguments passed to the underlying Hugging Face model's
                `save_pretrained` method.
        """
        if directory is None:
            directory = os.path.join(self.config.train.checkpoint_dir, "hf_model")

        self.accelerator.wait_for_everyone()

        # Save only the base model, so that is could be loaded directly
        # with Hugging Face's `from_pretrained` method
        state_dict = self.accelerator.get_state_dict(self.model, unwrap=True)

        self.accelerator.unwrap_model(self.model).save_pretrained(
            directory,
            save_function=self.accelerator.save,
            is_main_process=self.accelerator.is_main_process,
            state_dict=state_dict,
            **kwargs,
        )

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)
