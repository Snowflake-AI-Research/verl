from pathlib import Path
import torch
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.base import Worker
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from deepspeed.utils import OnDevice
from verl.utils import tensordict_utils as tu
from verl.utils import hf_tokenizer
import os
import ray
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from codetiming import Timer
import os
from contextlib import nullcontext
from functools import partial
from itertools import chain

import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh
import torch.nn.functional as F

try:
    from verl.workers.engine.mindspeed.transformer_impl import repatch
except ImportError:
    repatch = None
from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.flops_counter import FlopsCounter
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import ActorConfig, HFModelConfig, RolloutConfig, TrainingWorkerConfig
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import ppo_loss
from torch import Tensor
from verl.workers.engine.utils import postprocess_batch_func



def create_meta_model(name_or_path: str):
    model_config = AutoConfig.from_pretrained(name_or_path)
    with OnDevice(dtype=torch.float16, device='meta'):
        meta_model = AutoModelForCausalLM.from_config(model_config)
    return meta_model


def no_padding_2_padding_prompt_response(tensor: torch.Tensor, data: TensorDict, pad_token_id) -> torch.Tensor:
    """Convert jagged tensor into a left padded prompt and right padded prompt of [bsz, max_response_len], which looks like
    tensor([
      [pad...prompt | response...pad],
      [pad...prompt | response...pad],
      [pad...prompt | response...pad]
    ])

    Args:
        tensor: a nested tensor or a 1D tensor in shape (total_nnz,),
            total_nnz is the total number of tokens across all sequences in the batch
        data: TensorDict with "prompts", "responses", "attention_mask"
        pad_token_id: token to pad with

    Returns:
        tensor: sliced prompt+response tensor of shape [bsz, max_response_len] w/ left and right padding

    """
    # print(f"{tensor.is_nested=}")
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]
    # print(f"{prompt_ids.shape=}")
    # print(f"{response_ids.shape=}")
    # print(f"{attention_mask.shape=}")
    # print(f"{attention_mask=}")

    max_prompt_len = tu.get_non_tensor_data(data=data, key="max_prompt_len", default=-1)
    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    # print(f"data {max_prompt_len=}")
    # print(f"data {max_response_len=}")

    # print(f"{prompt_ids.is_nested=}")
    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_prompt_len < 0:
            max_prompt_len = prompt_lens.max().item()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_prompt_len = prompt_ids.shape[1]
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    # print(f"{data=}")
    # print(f"{prompt_lens=}")
    # print(f"{response_lens=}")
    # print(f"{max_prompt_len=}")
    # print(f"{max_response_len=}")
    # print(f"{sequence_offsets=}")
    # print(f"{values=}")
    # print(f"{values.shape=}")
    assert sequence_offsets[-1].item() == values.shape[0], f"{sequence_offsets[-1].item()} != {values.shape[0]}"

    input_ids_list = []
    for prompt_len, resp_len, seq_offset in zip(prompt_lens, response_lens, sequence_offsets, strict=True):
        prompt_pad_size = max_prompt_len - prompt_len
        response_pad_size = max_response_len - resp_len
        prompt = values[seq_offset - prompt_len - resp_len: seq_offset - resp_len]
        response = values[seq_offset - resp_len: seq_offset]
        prompt_padded_left = F.pad(prompt, (prompt_pad_size, 0), value=pad_token_id)
        response_padded_right = F.pad(response, (0, response_pad_size), value=pad_token_id)
        input_ids_list.append(torch.cat((prompt_padded_left, response_padded_right)))

    output = torch.stack(input_ids_list, dim=0)
    #print(f"{output=}")
    return output, max_prompt_len, max_response_len


def prepare_model_inputs_remove_padding(micro_batch: TensorDict):
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.utils.debug import log_gpu_memory_usage
    from verl.utils.device import get_device_id, get_device_name
    from verl.utils.model import extract_multi_modal_inputs
    from verl.utils.torch_functional import logprobs_from_logits
    import verl.utils.torch_functional as verl_F

    use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
    pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
    temperature = micro_batch["temperature"]
    temperature_item = temperature
    if use_fused_kernels:
        assert not isinstance(temperature, torch.Tensor), (
            "use_fused_kernels does not support per sample temperature yet"
        )
    assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

    multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
    input_ids = micro_batch["input_ids"]
    position_ids = micro_batch["position_ids"]

    if not isinstance(temperature, torch.Tensor):
        temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

    temperature = temperature.to(torch.float32)
    assert temperature.shape[0] == input_ids.shape[0]

    # args used to get outputs
    output_args = {}

    # support per sample temperature
    # temperature (bsz,)
    # input_ids (bsz, j1)
    temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()  # (total_nnz,)
    temperature_rmpad = temperature_rmpad.unsqueeze(0)  # (1, total_nnz)

    if pad_mode == DatasetPadMode.NO_PADDING:
        input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
        if position_ids.dim() == 3:
            position_ids_rmpad = position_ids.values().unsqueeze(1)  # (4, 1, total_nnz)
        else:
            position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
    else:
        raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

    # for compute the log_prob
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

    # pad and slice the inputs if sp > 1

    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
    temperature_rmpad = temperature_rmpad.squeeze(0)
    output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
    output_args["temperature_rmpad"] = temperature_rmpad

    # only pass input_ids and position_ids to enable flash_attn_varlen

    model_inputs = {
        "input_ids": input_ids_rmpad,
        "attention_mask": None,
        "position_ids": position_ids_rmpad,
        "labels": input_ids_rmpad,
    }

    extra_args = {}
    if use_fused_kernels:
        extra_args["temperature"] = temperature_item
        extra_args["return_dict"] = True

    model_inputs.update(multi_modal_inputs)
    model_inputs.update(extra_args)

    return model_inputs, output_args


def prepand_max_prompt_len_zeros(tensor: Tensor, max_prompt_len):
    prepand = torch.zeros([tensor.shape[0], max_prompt_len],  dtype=torch.int64, device=tensor.device)
    return torch.cat([prepand, tensor], dim=1)


def make_njt(data: TensorDict, tensor: Tensor) -> Tensor:
    cu_seqlens = data["input_ids"].offsets()
    seq_lengths = cu_seqlens.diff() # (bsz,)
    starts = torch.zeros_like(seq_lengths, dtype=torch.int64) # (bsz,)
    tensor = torch.nested.narrow(tensor, 1, starts, seq_lengths, layout=torch.jagged)
    tensor = torch.cat([t for t in tensor.unbind()])
    tensor = torch.nested.nested_tensor_from_jagged(tensor, cu_seqlens)
    return tensor


def prepare_padded_dss_batch_dict(data: TensorDict, pad_token_id) -> dict:
    input_ids = data['input_ids']
    position_ids = data['position_ids']

    orig_iput_ids_shape = input_ids.shape
    orig_position_ids_shape = position_ids.shape
    input_ids, max_prompt_len, max_response_len = no_padding_2_padding_prompt_response(tensor=input_ids, data=data, pad_token_id=pad_token_id)
    # XXX: 0 pad on pos ids is odd, check the original - perhaps need to re-build pos ids?
    position_ids, _, _= no_padding_2_padding_prompt_response(tensor=position_ids, data=data, pad_token_id=0)
    attention_mask = data['attention_mask']

    # print(f"{input_ids.shape=} {position_ids.shape=} {attention_mask.shape=} {orig_iput_ids_shape=} {orig_position_ids_shape=}")

    dss_batch_dict = dict(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

    return dss_batch_dict, max_prompt_len, max_response_len


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        self.role = role
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self.arctic_rl_client = kwargs.get("arctic_rl_client", None)
        assert self.arctic_rl_client is not None, "arctic_rl_client is required"

        self.use_zorro = ray.get(self.arctic_rl_client.is_zorro_enabled.remote())

        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=None, tool_config=None))

        if self._is_actor:
            model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            actor_training_config = TrainingWorkerConfig(
                model_type="language_model",
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )
            self.actor_config = actor_config

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.use_remove_padding

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None

            trust_remote_code=self.config.model.get("trust_remote_code", False)
            self.tokenizer = hf_tokenizer(self.config.model.path, trust_remote_code=trust_remote_code)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id

            self.device_name = get_device_name()
            self.flops_counter = FlopsCounter(model_config.hf_config)



    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
        self._register_dispatch_collect_info("ref", dp_rank=self.rank, is_collect=True)
        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)

        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def destroy(self):
        self.dss_training_engine.destroy()
        self.arctic_inference_engine.destroy()
        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        return


    def _update_config_params(self, data: TensorDict):
        default_keys = dict(
            use_remove_padding=self.config.model.use_remove_padding,
            use_dynamic_bsz=self.config.actor.use_dynamic_bsz,
            max_token_len_per_gpu=self.config.actor.ppo_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.config.actor.ppo_micro_batch_size_per_gpu,
            use_fused_kernels=self.config.actor.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})


    def compute_any_log_prob(self, data: TensorDict, compute_log_prob_fn) -> TensorDict:
        # print(f"compute_ref_log_prob data: {data}")
        batch, max_prompt_len, max_response_len = prepare_padded_dss_batch_dict(data, self.pad_token_id)

        self._update_config_params(data)

        #max_token_len_per_gpu = self.actor_config.ppo_max_token_len_per_gpu

        meta = dict(
            rollout_n=self.config.rollout.n,
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            max_token_len_per_gpu=data["max_token_len_per_gpu"],
            temperature=data["temperature"],
        )

        payload = dict(batch=batch, meta=meta)

        response = ray.get(compute_log_prob_fn.remote(payload))

        # print(f"compute_any_log_prob: {response['batch']['entropy'].shape=} {response['batch']['log_probs'].shape=}")

        #batch_output = postprocess_log_prob_output(data=data, entropy=entropy, log_probs=log_probs)
        #model_output = batch_output.pop("model_output", {})

        # verl wants a full [bs, max_prompt_len+max_response_len] tensors and jagged
        entropy = prepand_max_prompt_len_zeros(response['batch']['entropy'], max_prompt_len)
        log_probs = prepand_max_prompt_len_zeros(response['batch']['log_probs'], max_prompt_len)
        # print(f"compute_any_log_prob: {entropy.shape=} {log_probs.shape=}")
        entropy = make_njt(data, entropy)
        log_probs = make_njt(data, log_probs)
        # print(f"compute_any_log_prob: {entropy.shape=} {log_probs.shape=}")

        model_output = dict(entropy=entropy, log_probs=log_probs)
        metrics = response['metrics']
        # TODO: fix me - mfu is not computed here
        metrics["mfu"] = 0.0

        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": metrics})

        return final_output


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        return self.compute_any_log_prob(data, self.arctic_rl_client.compute_ref_log_prob)


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        return self.compute_any_log_prob(data, self.arctic_rl_client.compute_log_prob)


    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """
        # TODO: whether to log memory
        # metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024 ** 3)
        # metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024 ** 3)
        # metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        lr = metrics.pop("lr", None)

        final_metrics = metrics

        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_actor_global_batch(self, data: TensorDict) -> TensorDict:
        """Train a global batch

        Args:
            data:

        Returns:

        """
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)

        # update
        global_token_num = data["input_ids"].offsets().diff().tolist()  # (total_nnz,)
        tu.assign_non_tensor(
            data,
            global_token_num=NonTensorData(global_token_num),
            update_lr_scheduler=True,
            disable_auto_offload=disable_auto_offload,
        )

        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.config.model.use_remove_padding,
            use_dynamic_bsz=self.config.actor.use_dynamic_bsz,
            max_token_len_per_gpu=self.config.actor.ppo_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.config.actor.ppo_micro_batch_size_per_gpu,
            use_fused_kernels=self.config.actor.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            Timer(name="train_batch", logger=None) as timer,
        ):
            # XXX: what's missing is the loss function to be run on the dss side
            # arctic-verl/verl/workers/engine/fsdp/transformer_impl.py:1098 forward_step
            # the loss function is arctic-verl/verl/workers/utils/losses.py:97 ppo_loss
            # from verl.workers.utils.losses import ppo_loss <- need to adapt to pass a gazillion of config variables

            # from verl.utils.tensordict_utils import chunk_tensordict
            # batch = chunk_tensordict(data, 1)
            # print(f"update_actor data: {data}")

            # XXX: fix me
            input_ids = data['input_ids']
            position_ids = data['position_ids']
            #input_ids = input_ids.unbind()

            # XXX: move to init

            input_ids, max_prompt_len, max_response_len = no_padding_2_padding_prompt_response(tensor=input_ids, data=data, pad_token_id=self.pad_token_id)
            # XXX: 0 pad on pos ids is odd, check the original - perhaps need to re-build pos ids?
            position_ids, _, _= no_padding_2_padding_prompt_response(tensor=position_ids, data=data, pad_token_id=0)
            # print(f"{input_ids.shape=}")
            # print(f"{input_ids=}")

            #input_ids = torch.nested.to_padded_tensor(input_ids, padding=4.2)
            #position_ids = torch.nested.to_padded_tensor(position_ids, padding=4.2)

            # print(f"{data['attention_mask'].shape=}")
            # print(f"{data['attention_mask']=}")
            # print(f"{input_ids.shape=}")
            # print(f"{input_ids=}")
            # print(f"{position_ids.shape=}")
            # print(f"{position_ids=}")
            # XXX: fixme
            # batch = batch[0]

            #dss_batch_dict, output_args = prepare_model_inputs_remove_padding(data)
            # print(f"{dss_batch_dict=}")
            #print(f"{output_args=}")
            #import pdb; pdb.set_trace()

            batch = dict(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=data['attention_mask'],
                labels=input_ids,
                prompts=data["prompts"],
                responses=data["responses"],
                response_mask=data["response_mask"],
                old_log_probs=data["old_log_probs"],
                advantages=data["advantages"],
            )
            if self.config.actor.use_kl_loss:
                batch["ref_log_prob"] = data["ref_log_prob"]

            # print(f"{batch=}")

            # TODO: move to init since globally constant
            meta = dict(
                rollout_n=self.config.rollout.n,
                max_prompt_len=max_prompt_len,
                max_response_len=max_response_len,
                max_token_len_per_gpu=data["max_token_len_per_gpu"],
                temperature=data["temperature"],
                use_zorro=self.use_zorro,
                global_batch_size=data["global_batch_size"],
                rollout_is_weights=data.get("rollout_is_weights", None),
                batch_num_tokens=data["loss_mask"].sum(),
            )

            # we need to serialize the config object to dict
            # dataclasses.asdict only returns keys that are defined at init (vars will do more) - but perhaps we want `asdict`?
            actor_config_as_dict = vars(self.config.actor)
            # print(f"update_actor: {self.actor_config=}")
            # print(f"update_actor: {actor_config_as_dict=}")
            import json
            def safe_serialize(obj):
                return json.loads(json.dumps(obj, default=lambda o: None))
            actor_config_as_dict = safe_serialize(actor_config_as_dict)

            policy_loss_config = safe_serialize(vars(self.config.actor.policy_loss))

            meta.update(dict(actor_config=actor_config_as_dict, policy_loss_config=policy_loss_config))
            # print(f"update_actor: {post_process_inputs=}")


            payload = dict(batch=batch, meta=meta)
            response = ray.get(self.arctic_rl_client.update_actor.remote(payload))
            # output = ray.get(self.arctic_rl_client.update_actor.remote(dss_batch_dict, post_process_inputs))
            # print(f"update_actor: {loss=}")
            metrics = response['metrics']
            loss = metrics.pop("loss")
            # print(f"update_actor: {metrics=}")


        from verl.utils.metric import AggregationType, Metric
        # XXX: fix me - we need to aggregate the metrics
        metrics = {k:Metric(value=v[0] if isinstance(v, list) else v, aggregation=AggregationType.MEAN) for k,v in metrics.items()}
        metrics["lr"] = metrics.pop("last_lr")
        delta_time = timer.last

        # XXX: fix me
        # metrics = {
        #     'actor/pg_clipfrac': None,
        #     'actor/ppo_kl':  None,
        #     'actor/pg_clipfrac_lower':  None,
        #     'actor/pg_loss':  None,
        #     'kl_loss':  None,
        #     'kl_coef': None,
        #     'grad_norm': None,
        # }

        # print(f"{data=}")
        # print(f"{data["input_ids"].shape=}")

        # expected output so far
        #
        # output={
        # 'model_output': {
        #     'log_probs': NestedTensor(size=(1,j18), offsets=tensor([  0,401], device='cuda:0'), grad_fn=<NestedViewFromJaggedBackward0 object at 0x7f1cbc3630d0>, contiguous=True)
        # },
        # 'loss': [-0.9999991059303284],
        # 'metrics': {
        #     'actor/pg_clipfrac': <verl.utils.metric.utils.Metric object at 0x7f1e4e3faea0>,
        #     'actor/ppo_kl': <verl.utils.metric.utils.Metric object at 0x7f1e4e29fa70>,
        #     'actor/pg_clipfrac_lower': <verl.utils.metric.utils.Metric object at 0x7f1e4e29e0f0>,
        #     'actor/pg_loss': <verl.utils.metric.utils.Metric object at 0x7f1cbc21cd70>,
        #     'kl_loss': <verl.utils.metric.utils.Metric object at 0x7f1cbc21ce30>,
        #     'kl_coef': [0.001],
        #     'grad_norm': 16.321151733398438,
        #   }
        # }

        model_output = {}
        output = dict(
            model_output=model_output,
            metrics=metrics,
            loss=loss,
        )

        actor_output = self._postprocess_output(
            output,
            global_token_num=global_token_num,
            delta_time=delta_time,
            forward_only=False,
            images_seqlens=images_seqlens,
        ).cpu()

        output_metrics = tu.get(actor_output, "metrics")

        metrics = {}
        for key, val in output_metrics.items():
            # print(f"metrics {key=} {val=}")

            # flattn dp and micro batch
            if isinstance(val, list):
                output_metrics[key] = (
                    Metric.aggregate_dp(val)
                    if isinstance(val[0], Metric)
                    else list(chain.from_iterable(val))
                )

        append_to_dict(metrics, output_metrics)

        output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()

        return output


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: TensorDict) -> TensorDict:
        # output = self.actor.train_global_batch(data=data)
        output = self.train_actor_global_batch(data=data)
        return output.cpu() if output is not None else None

    # TODO: Load Checkpoint API
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        return


    # TODO: Save Checkpoint API
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        ray.get(self.arctic_rl_client.save_checkpoint.remote())
        return

    # TODO: Update Weights API
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.
        """
        ray.get(self.arctic_rl_client.update_weights.remote())
        return

    # TODO: CheckpointManager API Begin
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def sleep_replicas(self):
        """Sleep all rollout replicas: free weight and kv_cache device memory."""
        return
    # TODO: CheckpointManager API