#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
# BSZ=1024
BSZ=2
MBS=2
UBS=2
ROLL_N=2
MAX_STEPS=4
# LR=0 
LR=1e-6
# LOGGER=console
LOGGER="['console','wandb']"
USE_KL_LOSS=True
# USE_KL_LOSS=False 
# REMOVE_PADDING=True
REMOVE_PADDING=False
MODEL="Qwen/Qwen3-0.6B"
# STRATEGY="fsdp"
STRATEGY="fsdp2"
PYTHONUNBUFFERED=1 
HYDRA_FULL_ERROR=1
USE_LEGACY_WORKER_IMPL=disable
NGPU_PER_NODE=1
ROLLOUT_NAME=vllm

experiment_name="qwen3-0.6B_ngpu${NGPU_PER_NODE}_gbs${BSZ}_rolln${ROLL_N}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/code/shared/gsm8k/train.parquet \
    data.val_files=/code/shared/gsm8k/test.parquet \
    data.train_batch_size=${BSZ} \
    data.max_prompt_length=64 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.seed=42 \
    +actor_rollout_ref.actor.data_loader_seed=42 \
    reward.num_workers=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=${REMOVE_PADDING} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MBS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${UBS} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=${STRATEGY} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${UBS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=${ROLL_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${UBS} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=${STRATEGY} \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=${USE_LEGACY_WORKER_IMPL} \
    trainer.critic_warmup=0 \
    trainer.logger=${LOGGER} \
    trainer.experiment_name=${experiment_name} \
    trainer.project_name='verl_arctic_grpo_gsm8k' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NGPU_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=${MAX_STEPS} \
    trainer.total_epochs=15 $@ 2>&1 | tee ${experiment_name}.log

        # trainer.total_training_steps=${MAX_STEPS} \
