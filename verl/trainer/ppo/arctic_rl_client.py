import os
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
    cls = ArcticRLClientWrapper if USE_ARCTIC_TRAINING_CLIENT else ArcticRLClient4VeRL
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
        self.use_zorro = USE_ARCTIC_ZORRO
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
            entropy = torch.tensor(entropy).squeeze()
        if log_probs is not None:
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
        self.use_zorro = USE_ARCTIC_ZORRO

    def is_zorro_enabled(self):
        return self.use_zorro

    def initialize(self, model_name: str):
        from arctic_training.arctic_rl import ArcticRLClient, ArcticRLClientConfig

        config = ArcticRLClientConfig(
            host="localhost",
            port=7000,
            backend="local",
            # TODO: Grab GPU counts from VeRL config
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
            # TODO: Grab training config from VeRL config
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

        # Feels like a hack, but ArcticRLClient is constructed as a ray remote
        # actor with num_gpus=0 - This causes CUDA_VISIBLE_DEVICES to be empty,
        # so we need to set it manually.
        num_gpus = config.training_gpus + config.sample_gpus + config.log_prob_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

        self._client = ArcticRLClient(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # TODO: Just for debugging - remove later
    _default_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 1024,
    }

    def generate(self, prompt_ids, sampling_params) -> list:
        prompts = [self.tokenizer.decode(prompt_ids)] # TODO: pass prompt_ids directly
        merged_params = {**self._default_sampling_params, **sampling_params}
        return self._client.generate(prompts=prompts, sampling_params=merged_params)

    def compute_ref_log_prob(self, dss_batch_dict: dict, post_process_inputs: dict = None):
        return self.compute_log_prob(dss_batch_dict, post_process_inputs)

    def compute_log_prob(self, dss_batch_dict: dict, post_process_inputs: dict = None):
        batch = {
            "kwargs": dss_batch_dict,
            "context": {"labels": dss_batch_dict["labels"]},
        }
        result = self._client.fwd_no_grad(batch, post_processors=["entropy_logprobs"])
        outputs = result.get("model_outputs", result)

        entropy = outputs.get("entropy")
        log_probs = outputs.get("log_probs")

        if entropy is not None:
            entropy = torch.tensor(entropy)
        if log_probs is not None:
            log_probs = torch.tensor(log_probs)

        # TODO: AI fix to resolve problems after merge into arl branch. Need to verify correctness of this.
        # The model returns packed (1, total_nnz) tensors, but downstream
        # rm_padding expects padded (bsz, max_seq_len). Unpack and re-pad.
        if post_process_inputs is not None:
            cu_seqlens = post_process_inputs["cu_seqlens"]
            seq_lengths = cu_seqlens.diff()
            bsz = seq_lengths.shape[0]
            max_seq_len = int(seq_lengths.max())

            def _packed_to_padded(t):
                flat = t.reshape(-1)
                padded = torch.zeros(bsz, max_seq_len, dtype=flat.dtype, device=flat.device)
                for i in range(bsz):
                    start = int(cu_seqlens[i])
                    length = int(seq_lengths[i])
                    padded[i, :length] = flat[start:start + length]
                return padded

            if entropy is not None:
                entropy = _packed_to_padded(entropy)
            if log_probs is not None:
                log_probs = _packed_to_padded(log_probs)

        return entropy, log_probs

    def update_actor(self, dss_batch_dict: dict, post_process_inputs: dict):
        # TODO: Does this align with the ArcticRLClient4VeRL + dss-platform:verl_integration branch?
        extra = post_process_inputs.get("extra_inputs", {})
        seq_len = dss_batch_dict["input_ids"].shape[-1]

        def _left_pad(t: torch.Tensor) -> torch.Tensor:
            """Left-pad a response-only tensor to full sequence length with zeros."""
            pad_len = seq_len - t.shape[-1]
            if pad_len <= 0:
                return t
            pad = torch.zeros(*t.shape[:-1], pad_len, dtype=t.dtype, device=t.device)
            return torch.cat([pad, t], dim=-1)

        context = {
            "labels": dss_batch_dict["labels"],
            "old_logprobs": _left_pad(extra["old_log_probs"]),
            "advantages": _left_pad(extra["advantages"]),
            "loss_mask": _left_pad(extra["response_mask"]),
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
