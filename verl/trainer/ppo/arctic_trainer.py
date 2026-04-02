import torch
from typing import Optional
from torch.utils.data import Dataset, Sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.workers.arctic_workers import ActorRolloutRefWorker
from verl.trainer.ppo.utils import Role, WorkerType
from omegaconf import OmegaConf
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.arctic_rl_client import create_arctic_rl_client

def my_pdb():
    return 
    import pdb; pdb.set_trace()

class ArcticPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):        
        super().__init__(config=config, 
        tokenizer=tokenizer, 
        processor=processor, 
        role_worker_mapping=role_worker_mapping, 
        resource_pool_manager=resource_pool_manager, 
        ray_worker_group_cls=ray_worker_group_cls, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        collate_fn=collate_fn, 
        train_sampler=train_sampler, 
        device_name=device_name)

        self.use_gpu = False
        self.rl_client = create_arctic_rl_client()
        self.rl_client.initialize.remote(model_name="Qwen/Qwen3-0.6B")
        self.wg_kwargs["arctic_rl_client"] = self.rl_client
        

    def init_workers(self):
        super().init_workers()
        return 
        # print(f"ArcticPPOTrainer.init_workers: {self.actor_rollout_wg=}")
        # print(f"ArcticPPOTrainer.init_workers: {self.ref_policy_wg=}")
        # print(f"ArcticPPOTrainer.init_workers: {self.async_rollout_manager=}")
        # print(f"ArcticPPOTrainer.init_workers: {self.reward_loop_manager=}")
        # print(f"ArcticPPOTrainer.init_workers: {self.checkpoint_manager=}")

        # self.resource_pool_manager.create_resource_pool(use_gpu=self.use_gpu)

        # self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # # create actor and rollout
        # actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        # actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        # actor_rollout_cls = RayClassWithInitArgs(
        #     cls=self.role_worker_mapping[actor_role],
        #     config=self.config.actor_rollout_ref,
        #     role=str(actor_role),
        # )
        # self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # # create reference policy if needed
        # # if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
        # #     resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
        # #     ref_policy_cls = RayClassWithInitArgs(
        # #         self.role_worker_mapping[Role.RefPolicy],
        # #         config=self.config.actor_rollout_ref,
        # #         role=str(Role.RefPolicy),
        # #     )
        # #     self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # # initialize WorkerGroup
        # # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # # you should not use `create_colocated_worker_cls`.
        # # Instead, directly pass different resource pool to different worker groups.
        # # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        # all_wg = {}
        # wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        # if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
        #     wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        # if OmegaConf.select(self.config.global_profiler, "steps") is not None:
        #     wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
        #     # Only require nsight worker options when tool is nsys
        #     if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
        #         assert (
        #             OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
        #             is not None
        #         ), "worker_nsight_options must be set when using nsys with profile_steps"
        #         wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
        #             OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
        #         )
        # wg_kwargs["device_name"] = self.device_name

        # for resource_pool, class_dict in self.resource_pool_to_cls.items():
        #     if not class_dict:
        #         continue
        #     worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        #     wg_dict = self.ray_worker_group_cls(
        #         resource_pool=resource_pool,
        #         ray_cls_with_init=worker_dict_cls,
        #         use_gpu=self.use_gpu,
        #         **wg_kwargs,
        #     )
        #     spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        #     all_wg.update(spawn_wg)

 
        # self.actor_rollout_wg = all_wg[str(actor_role)]
        # self.actor_rollout_wg.init_model()

        # # create reward loop manager
        # from verl.experimental.reward_loop import RewardLoopManager

        # # initalize reward loop manager
        # # reward model (colocate or standalone): get resource_pool
        # # no reward model: resource_pool = None
        # resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        # self.reward_loop_manager = RewardLoopManager(
        #     config=self.config,
        #     rm_resource_pool=resource_pool,
        # )
        
        # self.async_rollout_mode = True
        # from verl.experimental.agent_loop import AgentLoopManager

        # # enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        # # reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        # # self.async_rollout_manager = AgentLoopManager.create(
        # #     config=self.config,
        # #     worker_group=self.actor_rollout_wg,
        # #     rollout_resource_pool=actor_rollout_resource_pool,
        # #     reward_loop_worker_handles=reward_loop_worker_handles,
        # # )

        # self.ref_policy_wg = self.actor_rollout_wg
        # self.checkpoint_manager = self.actor_rollout_wg
        # self.async_rollout_manager = self.actor_rollout_wg



    def destroy(self):
        # self.actor_rollout_wg.destroy() 
        self.rl_client.destroy.remote()