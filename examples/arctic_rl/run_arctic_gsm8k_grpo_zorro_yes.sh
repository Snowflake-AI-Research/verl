#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export HF_HOME=/checkpoint/huggingface
export USE_ARCTIC_TRAINING_CLIENT=1
# we want to make sure this runs on non-gpu client
export CUDA_VISIBLE_DEVICES=

BSZ=16
UBS=16
ROLL_N=5
MAX_STEPS=40
PROMPT_LENGTH=512
RESPONSE_LENGTH=1024

# BSZ=1
# UBS=1
# ROLL_N=16
# MAX_STEPS=4
# PROMPT_LENGTH=1024
# RESPONSE_LENGTH=2048

# BSZ=2
# UBS=2
# ROLL_N=4
# MAX_STEPS=4
# PROMPT_LENGTH=64
# RESPONSE_LENGTH=512

# LR=0
LR=1e-6

#LOGGER=console
LOGGER="['console','wandb']"
# USE_KL_LOSS=True
USE_KL_LOSS=False
# REMOVE_PADDING=True
REMOVE_PADDING=False
MODEL="Qwen/Qwen3-0.6B"
# STRATEGY="fsdp"
STRATEGY="fsdp2"
USE_LEGACY_WORKER_IMPL=disable
NGPU_PER_NODE=1
ROLLOUT_NAME=arctic # entry point into ArcticRL
USE_ARCTIC_RL=True
USE_ARCTIC_ZORRO=True
COLOCATE=False
experiment_name="qwen3-0.6B_ngpu${NGPU_PER_NODE}_gbs${BSZ}_rolln${ROLL_N}_zorro${USE_ARCTIC_ZORRO}"

gpu_name=$(nvidia-smi --query-gpu=gpu_name  --format=csv,noheader -i 0)
if [[ $gpu_name == *"H200"* ]]; then
    echo "Running on Hopper"
    flash_attention_v=flash_attention_3
elif [[ $gpu_name == *"B200"* ]] || [[ $gpu_name == *"B300"* ]] ; then
    echo "Running on Blackwell"
    flash_attention_v=flash_attention_2
else
    echo "Running on unknown: $gpu_name; don't know which FA version to use"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/code/shared/gsm8k/train.parquet \
    data.val_files=/code/shared/gsm8k/test.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=$PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    reward.num_workers=1 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=$REMOVE_PADDING \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$flash_attention_v \
    actor_rollout_ref.actor.strategy=$STRATEGY \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=$ROLL_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=$STRATEGY \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=$USE_LEGACY_WORKER_IMPL \
    trainer.use_arctic_rl=$USE_ARCTIC_RL \
    arctic_rl.colocate=$COLOCATE \
    arctic_rl.training_gpus=1\
    arctic_rl.sampling_gpus=2\
    arctic_rl.log_prob_gpus=1\
    arctic_rl.use_zorro=$USE_ARCTIC_ZORRO \
    trainer.critic_warmup=0 \
    trainer.logger=$LOGGER \
    trainer.experiment_name=$experiment_name \
    trainer.project_name=arctic_rl_bird_sql \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPU_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.total_epochs=15 \
    "$@" 2>&1 | tee $experiment_name.log

