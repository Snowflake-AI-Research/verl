import io
import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from deepspeed.utils import OnDevice
from dss_client.client import DSSInferenceClient, DSSTrainingClient, DSSLogProbClient
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group
from verl.workers.rollout.replica import TokenOutput
from tensordict import TensorDict
from typing import Any
from verl.utils.ray_utils import auto_await

USE_ARCTIC_TRAINING_CLIENT = os.environ.get("USE_ARCTIC_TRAINING_CLIENT", "0") == "1"


def create_arctic_rl_client():
    cls = ArcticRLClientWrapper if USE_ARCTIC_TRAINING_CLIENT else ArcticRLClient4VeRL
    sched_pg = placement_group([{"GPU": 0, "CPU": 1}])
    return ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=sched_pg,
            placement_group_capture_child_tasks=True,
        ),
    )(cls).remote()

def create_meta_model(name_or_path: str):
    model_config = AutoConfig.from_pretrained(name_or_path)
    with OnDevice(dtype=torch.float16, device='meta'):
        meta_model = AutoModelForCausalLM.from_config(model_config)
    return meta_model

class ArcticRLClient4VeRL:
    def __init__(self):
        self.arctic_inference_client = DSSInferenceClient(dss_server_url="http://localhost:7000")
        self.arctic_training_client = DSSTrainingClient(dss_server_url="http://localhost:7000")
        self.arctic_log_prob_client = DSSLogProbClient(dss_server_url="http://localhost:7000")

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
            "attn_implementation": "eager",
        }

        self.training_engine = self.arctic_training_client.initialize(
            model=create_meta_model(model_name),
            ds_config=ds_config,
            training_config=training_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt_ids, sampling_params) -> TokenOutput:
        prompts = [self.tokenizer.decode(prompt_ids)]
        result = self.inference_engine.generate(
            prompts=prompts,
        )
        return result

    def compute_log_prob(self, dss_batch_dict: dict):
        # XXX: somehow we need to differentiate which model is this called on ref vs actor - at the moment it's always actor hardcoded
        entropy, log_probs = self.training_engine.fwd_no_grad(**dss_batch_dict)

        if entropy is not None:
            # prior_entropy_shape = entropy.shape
            entropy = torch.tensor(entropy).squeeze()
        if log_probs is not None:
            # prior_log_probs_shape = log_probs.shape
            log_probs = torch.tensor(log_probs).squeeze()
        return entropy, log_probs


    def update_actor(self, dss_batch_dict: dict, post_process_inputs: dict):
        dss_batch_dict.update(post_process_inputs=post_process_inputs)

        _ = self.training_engine.forward(**dss_batch_dict)
        loss, metrics = self.training_engine.backward()
        self.training_engine.step()

        return loss.cpu().item(), metrics

    def destroy(self):
        self.training_engine.destroy()
        self.inference_engine.destroy()
        return


class ArcticRLClientWrapper:
    """Thin wrapper around ArcticTraining's ArcticRLClient that exposes the
    same interface as ArcticRLClient4VeRL so it can be used as a drop-in
    replacement.

    Set USE_ARCTIC_TRAINING_CLIENT=1 env var to activate.
    """

    def __init__(self):
        self._client = None
        self.tokenizer = None

    def initialize(self, model_name: str):
        from arctic_training.arctic_rl import ArcticRLClient, ArcticRLClientConfig

        config = ArcticRLClientConfig(
            host="localhost",
            port=7000,
            backend="local",
            training_gpus=1,
            sample_gpus=1,
            log_prob_gpus=1,
            log_prob_engine="deepspeed",
            model_name=model_name,
            ds_config={
                "train_micro_batch_size_per_gpu": 1,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "sequence_parallel_size": 1,
                "zero_optimization": {"stage": 1},
            },
            training_config={
                "optimizer": {"lr": 0.0002, "weight_decay": 0.0, "betas": [0.9, 0.999]},
                "lr_scheduler": {"warmup_ratio": 0.05},
                "training_horizon": 10,
                "max_length": 8096,
                "model_config": None,
                "attn_implementation": "eager",
            },
            vllm_config=None,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        self._client = ArcticRLClient(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    _default_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 1024,
    }

    def generate(self, prompt_ids, sampling_params) -> list:
        prompts = [self.tokenizer.decode(prompt_ids)]
        if sampling_params is not None and not isinstance(sampling_params, dict):
            merged_params = {**self._default_sampling_params, **vars(sampling_params)}
        else:
            merged_params = {**self._default_sampling_params, **(sampling_params or {})}
        return self._client.generate(prompts=prompts, sampling_params=merged_params)

    def compute_log_prob(self, dss_batch_dict: dict):
        batch = {
            "kwargs": dss_batch_dict,
            "context": {"labels": dss_batch_dict["labels"]},
        }
        result = self._client.fwd_no_grad(batch, post_processors=["entropy_logprobs"])
        outputs = result.get("model_outputs", result)

        entropy = outputs.get("entropy")
        log_probs = outputs.get("log_probs")

        if entropy is not None:
            entropy = torch.tensor(entropy).squeeze()
        if log_probs is not None:
            log_probs = torch.tensor(log_probs).squeeze()

        return entropy, log_probs

    def update_actor(self, dss_batch_dict: dict, post_process_inputs: dict):
        extra = post_process_inputs.get("extra_inputs", {})
        context = {
            "labels": dss_batch_dict["labels"],
            "old_logprobs": extra["old_log_probs"],
            "advantages": extra["advantages"],
            "loss_mask": extra["response_mask"],
        }
        batch = {"kwargs": dss_batch_dict, "context": context}

        result = self._client.fwd_bwd(batch, loss_fn="grpo")
        self._client.step()

        loss = result.get("avg_loss", 0.0)
        raw_metrics = result.get("post_process_outputs", {})
        # Caller expects metrics values to be lists (does v[0])
        metrics = {k: v if isinstance(v, list) else [v] for k, v in raw_metrics.items()}

        return loss, metrics

    def destroy(self):
        if self._client is not None:
            self._client.shutdown()
