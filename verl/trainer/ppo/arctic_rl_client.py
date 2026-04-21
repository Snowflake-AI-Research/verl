import os
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from deepspeed.utils import OnDevice
from dss_client.client import DSSInferenceClient, DSSTrainingClient, DSSLogProbClient
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group
from verl.workers.rollout.replica import TokenOutput

USE_ARCTIC_TRAINING_CLIENT = os.environ.get("USE_ARCTIC_TRAINING_CLIENT", "0") == "1"
USE_ARCTIC_ZORRO = os.environ.get("USE_ARCTIC_ZORRO", "0") == "1"


def create_arctic_rl_client(config):
    cls = ArcticRLClientWrapper if config.arctic_rl.use_arctic_rl_client else ArcticRLClient4VeRL
    sched_pg = placement_group([{"GPU": 0, "CPU": 1}])
    return ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=sched_pg,
            placement_group_capture_child_tasks=True,
        ),
    )(cls).remote(config)

def create_meta_model(name_or_path: str):
    model_config = AutoConfig.from_pretrained(name_or_path)
    with OnDevice(dtype=torch.float16, device='meta'):
        meta_model = AutoModelForCausalLM.from_config(model_config)
    return meta_model


class ArcticRLClient4VeRL:
    def __init__(self, config):
        """
        config: verl's full config
        """
        self.config = config
        self.use_zorro = self.config.arctic_rl.use_zorro
        #print(f"ArcticRLClient4VeRL {config=}")

        self.arctic_inference_client = DSSInferenceClient(dss_server_url="http://localhost:7000")
        self.arctic_training_client = DSSTrainingClient(dss_server_url="http://localhost:7000")
        self.arctic_log_prob_client = DSSLogProbClient(dss_server_url="http://localhost:7000")


    def is_zorro_enabled(self):
        return self.use_zorro

    def initialize(self, model_name: str):
        vllm_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 1024,
            "stop_sequences": [],
            "stop_token_ids": [],
        }
        self.inference_engine = self.arctic_inference_client.initialize(
            model_name=model_name,
            vllm_config=vllm_config,
        )
        self.log_prob_engine = self.arctic_log_prob_client.initialize(
            model_name=model_name,
            vllm_config=vllm_config,
        )

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sequence_parallel_size": 1,
            "zero_optimization": {
                "stage": 1,
            },
        }

        # currently verl wants '+' before the setting, i.e. +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_3
        attn_implementation = self.config.actor_rollout_ref.model.override_config.get('attn_implementation', 'eager')
        if attn_implementation == "eager":
            raise ValueError("set actor_rollout_ref.model.override_config.attn_implementation to some variant of flash attention")

        #attn_implementation="flash_attention_3"

        training_config = {
            "optimizer": {
                "lr": 0.0002,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "lr_scheduler": {"warmup_ratio": 0.05},
            "training_horizon": 10,
            "max_length": 8096,
            "model_config": None,
            "attn_implementation": attn_implementation,
        }

        if self.is_zorro_enabled():
            # XXX: can't find where it's configured
            use_unpad = True

            training_config.update(
                use_zorro=True,
                response_len=self.config.data.max_response_length,
                max_token_len=self.config.actor_rollout_ref.rollout.max_num_batched_tokens,
                rollout_n=self.config.actor_rollout_ref.rollout.n,
                temperature=self.config.actor_rollout_ref.rollout.temperature,
                use_unpad=use_unpad,
            )
            #print(f"{training_config=}")

        self.training_engine = self.arctic_training_client.initialize(
            model=create_meta_model(model_name),
            ds_config=ds_config,
            training_config=training_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt_ids, sampling_params) -> TokenOutput:
        prompts = [self.tokenizer.decode(prompt_ids)]
        return self.inference_engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )


    # TODO: this should use the reference engine instead of the training engine
    def compute_ref_log_prob(self, payload: dict):
        response = self.training_engine.fwd_no_grad(**payload)
        # if entropy is not None:
        #     entropy = torch.tensor(entropy).squeeze()
        # if log_probs is not None:
        #     log_probs = torch.tensor(log_probs).squeeze()
        print(f"arctic_rl_client.compute_ref_log_prob: {response['batch']['entropy'].shape=}, {response['batch']['log_probs'].shape=}")
        return response


    def compute_log_prob(self, payload: dict):
        # XXX: somehow we need to differentiate which model is this called on ref vs actor - at the moment it's always actor hardcoded
        response = self.training_engine.fwd_no_grad(**payload)

        # XXX: for some reason no_padding_2_padding expects a 1D tensor - not sure how it'll work for
        # bs>1
        # I think it may have to do with tensor.is_nested - different path/logic
        # so most likely we need to convert these 2 into TensorDict
        # if entropy is not None:
        #     entropy = torch.tensor(entropy).squeeze()
        # if log_probs is not None:
        #     log_probs = torch.tensor(log_probs).squeeze()
        print(f"arctic_rl_client.compute_log_prob: {response['batch']['entropy'].shape=}, {response['batch']['log_probs'].shape=}")
        return response


    def update_actor(self, payload: dict):
        _ = self.training_engine.forward(**payload)
        bwd_response = self.training_engine.backward()
        step_response = self.training_engine.step()

        step_response["metrics"].update(**bwd_response["metrics"])

        # metrics.update({"global_steps": [global_steps], "last_lr": [last_lr]})

        # print(f"arctic_rl_client.update_actor: {loss=}")
        # print(f"arctic_rl_client.update_actor: {metrics=}")
        # return loss.cpu().item(), metrics

        return step_response

    def destroy(self):
        self.training_engine.destroy()
        self.inference_engine.destroy()
        return

# TODO: Once we are happy with this implementation, we can make this the new
# ArcticRLClient4VeRL.
class ArcticRLClientWrapper:
    """Thin wrapper around ArcticTraining's ArcticRLClient that exposes the
    same interface as ArcticRLClient4VeRL so it can be used as a drop-in
    replacement.

    Set USE_ARCTIC_TRAINING_CLIENT=1 env var to activate.
    """

    def __init__(self, config):
        self.config = config
        self._client = None
        self.tokenizer = None
        self.use_zorro = self.config.arctic_rl.use_zorro

    def is_zorro_enabled(self):
        return self.use_zorro

    def _create_ds_config(self, n_gpus: int) -> dict[str, Any]:
        actor_cfg = self.config.actor_rollout_ref.actor
        data_cfg = self.config.data

        micro_batch_size = actor_cfg.ppo_micro_batch_size_per_gpu or 1
        train_batch_size = data_cfg.train_batch_size * self.config.actor_rollout_ref.rollout.n
        grad_accum_steps = max(1, train_batch_size // (micro_batch_size * n_gpus))
        train_seq_parallel_size = actor_cfg.fsdp_config.get("ulysses_sequence_parallel_size", 1)
        return {
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "sequence_parallel_size": train_seq_parallel_size,
            "zero_optimization": {"stage": 1},
    }

    def _create_ds_worker_config(self):

        if self.is_zorro_enabled():
            # XXX: can't find where it's configured
            use_unpad = True

            return dict(
                use_zorro=True,
                response_len=self.config.data.max_response_length,
                max_token_len=self.config.actor_rollout_ref.rollout.max_num_batched_tokens,
                rollout_n=self.config.actor_rollout_ref.rollout.n,
                temperature=self.config.actor_rollout_ref.rollout.temperature,
                use_unpad=use_unpad,
            )
        else:
            return {}


    def initialize(self, model_name: str):
        from arctic_training.arctic_rl import ArcticRLClient, ArcticRLClientConfig

        n_training_gpus = self.config.arctic_rl.get("training_gpus", self.config.trainer.n_gpus_per_node)
        n_sampling_gpus = self.config.arctic_rl.get("sampling_gpus", self.config.trainer.n_gpus_per_node)
        n_log_prob_gpus = self.config.arctic_rl.get("log_prob_gpus", self.config.trainer.n_gpus_per_node)
        colocate = self.config.arctic_rl.get("colocate", False)
        attn_implementation = self.config.actor_rollout_ref.model.override_config.get(
            'attn_implementation', 'eager'
        )

        actor_cfg = self.config.actor_rollout_ref.actor
        optim_cfg = actor_cfg.optim
        data_cfg = self.config.data

        max_length = data_cfg.max_prompt_length + data_cfg.max_response_length

        rollout_cfg = self.config.actor_rollout_ref.rollout
        vllm_config = {
            "tensor_parallel_size": rollout_cfg.tensor_model_parallel_size,
            "gpu_memory_utilization": rollout_cfg.gpu_memory_utilization,
            "max_model_len": rollout_cfg.get("max_model_len") or max_length,
            "max_num_seqs": rollout_cfg.max_num_seqs,
            "enforce_eager": rollout_cfg.enforce_eager,
            "enable_chunked_prefill": rollout_cfg.enable_chunked_prefill,
        }
        if rollout_cfg.get("quantization"):
            vllm_config["quantization"] = rollout_cfg.quantization

        rl_config = ArcticRLClientConfig(
            host="localhost",
            port=7000,
            backend="local",
            training_gpus=n_training_gpus,
            sampling_gpus=n_sampling_gpus,
            log_prob_gpus=n_log_prob_gpus,
            colocate=colocate,
            log_prob_engine="deepspeed",
            model_name=model_name,
            ds_config=self._create_ds_config(n_training_gpus),
            log_prob_ds_config=self._create_ds_config(n_log_prob_gpus),
            training_config={
                "optimizer": {
                    "lr": optim_cfg.lr,
                    "weight_decay": optim_cfg.weight_decay,
                    "betas": list(optim_cfg.betas),
                },
                "lr_scheduler": {"warmup_ratio": optim_cfg.lr_warmup_steps_ratio},
                "training_horizon": self.config.trainer.total_epochs,
                "max_length": max_length,
                "model_config": None,
                "attn_implementation": attn_implementation,
            },
            ds_worker_config=self._create_ds_worker_config(),
            vllm_config=vllm_config,
        )

        # ArcticRLClient is constructed as a ray remote actor with num_gpus=0,
        # which causes CUDA_VISIBLE_DEVICES to be empty.
        if colocate:
            num_visible = n_training_gpus + n_sampling_gpus + n_log_prob_gpus
        else:
            num_visible = rl_config.training_gpus + rl_config.sampling_gpus + rl_config.log_prob_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_visible))

        self._client = ArcticRLClient(rl_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # TODO: Just for debugging - remove later
    _default_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 1024,
    }

    async def generate(self, prompt_ids, sampling_params) -> list:
        prompts = [self.tokenizer.decode(prompt_ids)] # TODO: pass prompt_ids directly
        merged_params = {**self._default_sampling_params, **sampling_params}
        return await self._client.async_generate(prompts=prompts, sampling_params=merged_params)


    def compute_ref_log_prob(self, payload: dict):
        payload["processing"] = {"post": ["compute_logprobs", "compute_entropy"], "loss_fn": None}
        response = self._client.fwd_no_grad(payload, reference_model=True)
        response["batch"]["log_probs"] = response["batch"].pop("logprobs")
        print(f"[ArcticRLWrapper] compute_ref_log_prob OUTPUT: {response.keys()=}")
        return response


    def compute_log_prob(self, payload: dict):
        payload["processing"] = {"post": ["compute_logprobs", "compute_entropy"], "loss_fn": None}
        response = self._client.fwd_no_grad(payload, reference_model=False)
        response["batch"]["log_probs"] = response["batch"].pop("logprobs")
        print(f"[ArcticRLWrapper] compute_log_prob OUTPUT: {response.keys()=}")
        return response

    def update_actor(self, payload: dict):
        payload["processing"] = {
            "post": ["apply_temperature", "compute_logprobs", "compute_entropy"],
            # "loss_fn": "verl_grpo"
            "loss_fn": "grpo"
        }
        def _left_pad(t: torch.Tensor, seq_len: int) -> torch.Tensor:
            """Left-pad a response-only tensor to full sequence length with zeros."""
            pad_len = seq_len - t.shape[-1]
            if pad_len <= 0:
                return t
            pad = torch.zeros(*t.shape[:-1], pad_len, dtype=t.dtype, device=t.device)
            return torch.cat([pad, t], dim=-1)

        seq_len = payload["batch"]["input_ids"].shape[-1]
        for name in ["old_log_probs", "advantages", "response_mask", "ref_log_prob"]:
            if name in payload["batch"]:
                payload["batch"][name] = _left_pad(payload["batch"][name], seq_len)

        payload["batch"]["loss_mask"] = payload["batch"]["response_mask"]

        fwd_bwd_response = self._client.fwd_bwd(payload)
        print(f"[ArcticRLWrapper] update_actor OUTPUT: {fwd_bwd_response.keys()=}")
        step_response = self._client.step()
        print(f"[ArcticRLWrapper] update_actor STEP OUTPUT: {step_response.keys()=}")
        step_response["metrics"].update(**fwd_bwd_response["metrics"])
        return step_response


    def save_checkpoint(self):
        response = self._client.save_checkpoint()
        print(f"[ArcticRLClientWrapper] save_checkpoint OUTPUT: {response.keys()=}")
        return response

    def update_weights(self):
        response = self._client.sync_weights()
        print(f"[ArcticRLClientWrapper] update_weights OUTPUT: {response.keys()=}")
        return response

    def destroy(self):
        if self._client is not None:
            self._client.shutdown()
