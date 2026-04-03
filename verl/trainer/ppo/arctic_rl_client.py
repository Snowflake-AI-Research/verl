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

def create_arctic_rl_client():
    sched_pg = placement_group([{"GPU": 0, "CPU": 1}])
    arctic_rl_client = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=sched_pg,
            placement_group_capture_child_tasks=True,
        ),
    )(ArcticRLClient4VeRL).remote(
    )

    return arctic_rl_client

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
        return self.inference_engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )

    # TODO: this should use the reference engine instead of the training engine
    def compute_ref_log_prob(self, dss_batch_dict: dict, post_process_inputs: dict):
        dss_batch_dict.update(post_process_inputs=post_process_inputs)
        entropy, log_probs = self.training_engine.fwd_no_grad(**dss_batch_dict)
        if entropy is not None:
            entropy = torch.tensor(entropy).squeeze()
        if log_probs is not None:
            log_probs = torch.tensor(log_probs).squeeze()
        print(f"arctic_rl_client.compute_ref_log_prob: {entropy.shape=}, {log_probs.shape=}")
        return entropy, log_probs
    

    def compute_log_prob(self, dss_batch_dict: dict, post_process_inputs: dict):
        dss_batch_dict.update(post_process_inputs=post_process_inputs)

        # XXX: somehow we need to differentiate which model is this called on ref vs actor - at the moment it's always actor hardcoded
        entropy, log_probs = self.training_engine.fwd_no_grad(**dss_batch_dict)

        # XXX: for some reason no_padding_2_padding expects a 1D tensor - not sure how it'll work for
        # bs>1
        # I think it may have to do with tensor.is_nested - different path/logic
        # so most likely we need to convert these 2 into TensorDict
        if entropy is not None:
            # prior_entropy_shape = entropy.shape
            entropy = torch.tensor(entropy).squeeze()
        if log_probs is not None:
            # prior_log_probs_shape = log_probs.shape
            log_probs = torch.tensor(log_probs).squeeze()
        print(f"arctic_rl_client.compute_log_prob: {entropy.shape=}, {log_probs.shape=}")
        return entropy, log_probs


    def update_actor(self, dss_batch_dict: dict, post_process_inputs: dict):

        dss_batch_dict.update(post_process_inputs=post_process_inputs)

        #_ = self.training_engine.forward(**dss_batch_dict, post_process_inputs=post_process_inputs)
        _ = self.training_engine.forward(**dss_batch_dict)
        loss, metrics = self.training_engine.backward()
        self.training_engine.step()

        print(f"arctic_rl_client.update_actor: {loss=}")
        print(f"arctic_rl_client.update_actor: {metrics=}")
        return loss.cpu().item(), metrics

    def destroy(self):
        self.training_engine.destroy()
        self.inference_engine.destroy()
        return

