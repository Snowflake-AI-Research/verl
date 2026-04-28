#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export HF_HOME=/checkpoint/huggingface
# we want to make sure this runs on non-gpu client
export CUDA_VISIBLE_DEVICES=

BSZ=1024
UBS=32
ROLL_N=5
PROMPT_LENGTH=512
RESPONSE_LENGTH=1024
MAX_STEPS=100


BSZ=8
UBS=2
ROLL_N=2
PROMPT_LENGTH=512
RESPONSE_LENGTH=1024
MAX_STEPS=4


experiment_name="qwen3-0.6B_arctic_gsm8k_grpo"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/code/shared/gsm8k/train.parquet \
    data.val_files=/code/shared/gsm8k/test.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    reward.num_workers=1 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_3 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=arctic \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=$ROLL_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.use_arctic_rl=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.experiment_name=$experiment_name \
    trainer.project_name=arctic_gsm8k_grpo \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=${MAX_STEPS} \
    arctic_rl.colocate=False \
    arctic_rl.training_gpus=2\
    arctic_rl.sampling_gpus=2\
    arctic_rl.log_prob_gpus=0\
    arctic_rl.use_zorro=True \
    "$@" 2>&1 | tee $experiment_name.log
