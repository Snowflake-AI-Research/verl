#!/bin/bash
# GRPO training for Qwen3-1.7B on BIRD SQL dataset
# Adapted from exp64_lr5e-6_416k (SnowflakeDialectSQLRewardManagerV6b config)
#
# 1 node, 8 GPUs
#
# Prerequisites:
#   1. Preprocess data:  python examples/bird_sql/preprocess_bird.py
#   2. pip install func_timeout

set -x

experiment_name='qwen3_1.7b_bird_grpo_zorro_no'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
MAX_STEPS=4
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export HF_HOME=/checkpoint/huggingface
export USE_ARCTIC_TRAINING_CLIENT=1
export CUDA_VISIBLE_DEVICES=
USE_ARCTIC_RL=True # entry point into ArcticRL

USE_LEGACY_WORKER_IMPL=disable
ROLLOUT_NAME=arctic
NUM_AGENT_WORKERS=1
NGPU_PER_NODE=1

# BSZ=128
# PROMPT_LEN=16384
# RESPONSE_LEN=4096
# ROLL_N=16

BSZ=2
PROMPT_LEN=16384
RESPONSE_LEN=4096
ROLL_N=2
# LOGGER=console
LOGGER="['console','wandb']"

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

# DATA_DIR="/data/snowflakesql/xyu/open-source-text2sql"
# TRAIN_FILES="${DATA_DIR}/train.parquet"
# VAL_FILES="${DATA_DIR}/val.parquet"

DATA_DIR="/code/shared/open-source-text2sql"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/val.parquet"


# LOG_PROBS=True
LOG_PROBS=False

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${BSZ} \
    data.max_prompt_length=${PROMPT_LEN} \
    data.max_response_length=${RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.truncation=left \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$flash_attention_v \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLL_N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=${LOG_PROBS} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=1800 \
    trainer.use_legacy_worker_impl=${USE_LEGACY_WORKER_IMPL} \
    trainer.use_arctic_rl=${USE_ARCTIC_RL} \
    trainer.balance_batch=False \
    trainer.default_local_dir=/data-fast/sql-rl/${experiment_name} \
    trainer.logger=${LOGGER} \
    trainer.project_name=arctic_rl_bird_sql \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPU_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False \
    custom_reward_function.path="${SCRIPT_DIR}/bird_reward.py" \
    custom_reward_function.name=compute_score \
    trainer.total_training_steps=${MAX_STEPS} \
    "$@" 2>&1 | tee ${experiment_name}.log
